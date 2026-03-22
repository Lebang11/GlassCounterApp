import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, Image, TouchableOpacity, ActivityIndicator, Alert, Dimensions } from 'react-native';
import { File, Directory } from 'expo-file-system';
import * as Sharing from 'expo-sharing';
import { Ionicons } from '@expo/vector-icons';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO, decodeJpeg } from '@tensorflow/tfjs-react-native';
import { Buffer } from 'buffer';

const VECTOR_SIZE = 1024;

// Load the model from the sharded assets
const modelJson = require('../assets/web_model/model.json');
const modelWeights = require('../assets/web_model/group1-shard1of1.bin');

// Horizontal Sobel kernel for vertical edge detection (detects vertical lines in X direction)
const SOBEL_X_KERNEL_3X3 = [-1, 0, 1, -2, 0, 2, -1, 0, 1];

export default function ResultScreen({ photo, onRetake }) {
    const [sheetCount, setSheetCount] = useState(null);
    const [isProcessing, setIsProcessing] = useState(true);
    const [model, setModel] = useState(null);

    useEffect(() => {
        loadModelAndRun();
    }, []);

    const loadModelAndRun = async () => {
        try {
            setIsProcessing(true);

            // 1. Initialize TFJS (already done in App.js usually, but safe here)
            await tf.ready();

            // 2. Load the Hybrid CNN Model
            // Using bundleResourceIO for Expo sharded assets
            const loadedModel = await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeights));
            setModel(loadedModel);

            // 3. Preprocess and Run Inference
            await runInference(loadedModel);
        } catch (error) {
            console.error(error);
            Alert.alert("Model Error", "Failed to load/run the ML model: " + error.message);
            setIsProcessing(false);
        }
    };

    const preprocessImage = async (uri) => {
        // Since we are doing 1D processing (Hybrid Fourier-CNN), 
        // we replicate the Python 'preprocess_image_1d' logic in JS.

        // Load image as tensor using the new SDK 54 File API
        const file = new File(uri);
        const imgB64 = await file.base64();

        // Use Buffer (from direct dependency of TFJS RN) for robust Base64 decoding
        const rawImageData = new Uint8Array(Buffer.from(imgB64, 'base64'));

        const imageTensor = decodeJpeg(rawImageData); // [H, W, 3]
        console.log("Image sensor loaded with shape:", imageTensor.shape);

        return tf.tidy(() => {
            // 1. Convert to normalized grayscale [0.0 - 1.0]
            let grayscale = imageTensor.mean(2).div(255).expandDims(0).expandDims(-1);

            // 2. Apply Horizontal Sobel Filter (dx=1) to detect Vertical Sheets
            const kernel = tf.tensor4d(SOBEL_X_KERNEL_3X3, [3, 3, 1, 1]).reverse([0, 1]);
            let edges = tf.conv2d(grayscale, kernel, 1, 'same').abs();

            // 3. Normalize intensity
            const maxVal = edges.max();
            console.log("Maximum Sobel Edge Strength:", maxVal.dataSync()[0]);
            edges = edges.div(maxVal.add(1e-8)).mul(255);

            const h = edges.shape[1];
            const w = edges.shape[2];

            // 4. Multi-strip average (Horizontal stripes for Vertical lines)
            const stripHeight = Math.floor(h / 30); // Use a narrower stripe to avoid blurring
            const strips = [];
            for (let i = 1; i <= 5; i++) {
                const centerH = Math.floor((h * i) / 6);
                const startY = Math.max(0, centerH - Math.floor(stripHeight / 2));
                const strip = edges.slice([0, startY, 0, 0], [1, stripHeight, w, 1]);
                const signal = strip.mean([0, 1, 3]); // Mean across Batch, Height, and Channel -> [W]
                strips.push(signal);
            }

            // 5. Average signals across stripes
            let avgSignal = tf.addN(strips).div(tf.scalar(5)); // [W]

            // 6. Resize full signal width to 1024 (No longer cropping background)
            let signal4d = avgSignal.reshape([1, 1, w, 1]);
            let resized = tf.image.resizeBilinear(signal4d, [1, VECTOR_SIZE]).reshape([VECTOR_SIZE]);

            // 8. Adaptive Normalization (Min-Max)
            const min = resized.min();
            const max = resized.max();
            let signalNormalized = resized.sub(min).div(max.sub(min).add(1e-8));

            // 9. Standardization (Z-score)
            const moments = tf.moments(signalNormalized);
            const mean = moments.mean.dataSync()[0];
            const variance = moments.variance.dataSync()[0];
            console.log("Signal Stats - Mean:", mean, "Var:", variance);

            const finalSignal = signalNormalized.sub(moments.mean).div(tf.sqrt(moments.variance.add(1e-8)));

            // Reshape for Conv1D input: [1, 1024, 1]
            return finalSignal.reshape([1, VECTOR_SIZE, 1]);
        });
    };

    const estimateFourierCount = (signalTensor) => {
        // Replicate 'estimate_count_fourier' logic from Python notebook
        return tf.tidy(() => {
            const signal = signalTensor.reshape([VECTOR_SIZE]);
            const centered = signal.sub(signal.mean());

            // Compute real-valued FFT magnitude
            const fft = tf.spectral.rfft(centered).abs();

            // Find peak frequency in the glass sheet range (indices 10 to 100)
            const fftSection = fft.slice([10], [90]);
            const peakIdxOffset = fftSection.argMax().dataSync()[0];
            const peakIdx = peakIdxOffset + 10;

            // Fourier estimated count
            const count = (peakIdx / VECTOR_SIZE) * VECTOR_SIZE;
            console.log("Fourier Density Estimate - Peak Index:", peakIdx, "Frequency Score:", count);
            return Math.round(count);
        });
    };

    const runInference = async (loadedModel) => {
        try {
            const inputTensor = await preprocessImage(photo.uri);
            const prediction = loadedModel.predict(inputTensor);
            const rawValue = prediction.dataSync()[0];
            const cnnCount = Math.round(rawValue);

            const fourierCount = estimateFourierCount(inputTensor);

            console.log("ML Prediction Raw Result:", rawValue);
            console.log("CNN Count:", cnnCount, "| Fourier Count:", fourierCount);

            // Final count is a weighted consensus (Hybrid).
            // Usually we trust the CNN more, but Fourier is a great sanity check.
            const finalCount = Math.abs(cnnCount - fourierCount) < 5
                ? Math.round((cnnCount + fourierCount) / 2)
                : cnnCount;

            setSheetCount(fourierCount);
            setIsProcessing(false);

            // Cleanup tensors
            inputTensor.dispose();
            prediction.dispose();
        } catch (error) {
            console.error("Inference failed", error);
            Alert.alert("Inference Error", "Failed to analyze image: " + error.message);
            setIsProcessing(false);
        }
    };

    const exportResultParams = async () => {
        if (sheetCount === null) return;
        const csvContent = `image_number,number_of_sheets\n1,${sheetCount}`;
        const resultsFile = new File(Directory.document, "results.csv");

        try {
            await resultsFile.write(csvContent);
            if (await Sharing.isAvailableAsync()) {
                await Sharing.shareAsync(resultsFile.uri);
            } else {
                Alert.alert("Error", "Sharing not available on this device.");
            }
        } catch (e) {
            Alert.alert("Error", e.message);
        }
    };

    return (
        <View style={styles.container}>
            <Image source={{ uri: photo.uri }} style={styles.imageBackground} blurRadius={isProcessing ? 10 : 0} />
            <View style={[styles.darkOverlay, { opacity: isProcessing ? 0.7 : 0.4 }]} />

            <View style={styles.overlay}>
                <View style={styles.resultCard}>
                    <Text style={styles.title}>Glass Stack Count</Text>

                    {isProcessing ? (
                        <View style={styles.loadingContainer}>
                            <ActivityIndicator size="large" color="#00ffcc" />
                            <Text style={styles.loadingText}>Analyzing image tensors...</Text>
                        </View>
                    ) : (
                        <View style={styles.countContainer}>
                            <Text style={styles.countNumber}>{sheetCount}</Text>
                            <Text style={styles.countLabel}>Sheets</Text>
                            <Text style={styles.maeText}>Model: Fourier</Text>
                        </View>
                    )}

                    <View style={styles.actionRow}>
                        <TouchableOpacity style={styles.actionButton} onPress={onRetake}>
                            <Ionicons name="camera-reverse" size={24} color="#fff" />
                            <Text style={styles.actionText}>Retake</Text>
                        </TouchableOpacity>

                        {!isProcessing && (
                            <TouchableOpacity style={[styles.actionButton, styles.primaryButton]} onPress={exportResultParams}>
                                <Ionicons name="document-text" size={24} color="#000" />
                                <Text style={[styles.actionText, { color: '#000' }]}>Save CSV</Text>
                            </TouchableOpacity>
                        )}
                    </View>
                </View>
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#000',
    },
    imageBackground: {
        position: 'absolute',
        width: '100%',
        height: '100%',
    },
    darkOverlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'black',
    },
    overlay: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
    },
    resultCard: {
        backgroundColor: 'rgba(15, 23, 42, 0.90)',
        borderRadius: 24,
        padding: 30,
        width: '100%',
        alignItems: 'center',
        borderWidth: 1,
        borderColor: 'rgba(0, 255, 204, 0.4)',
        shadowColor: '#00ffcc',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.2,
        shadowRadius: 20,
        elevation: 10,
    },
    title: {
        color: '#fff',
        fontSize: 24,
        fontWeight: '800',
        marginBottom: 20,
    },
    loadingContainer: {
        alignItems: 'center',
        marginVertical: 40,
    },
    loadingText: {
        color: '#00ffcc',
        marginTop: 20,
        fontSize: 16,
        fontWeight: '600',
    },
    countContainer: {
        alignItems: 'center',
        marginVertical: 10,
    },
    countNumber: {
        fontSize: 80,
        fontWeight: '900',
        color: '#00ffcc',
        textShadowColor: 'rgba(0, 255, 204, 0.5)',
        textShadowOffset: { width: 0, height: 0 },
        textShadowRadius: 15,
    },
    countLabel: {
        fontSize: 20,
        fontWeight: '700',
        color: '#94a3b8',
        textTransform: 'uppercase',
        letterSpacing: 3,
        marginTop: 5,
    },
    maeText: {
        fontSize: 14,
        fontWeight: 'bold',
        color: '#00ffcc',
        opacity: 0.8,
        marginTop: 15,
    },
    actionRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        width: '100%',
        marginTop: 40,
    },
    actionButton: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(255, 255, 255, 0.1)',
        paddingVertical: 14,
        paddingHorizontal: 20,
        borderRadius: 16,
        flex: 1,
        marginHorizontal: 5,
        justifyContent: 'center'
    },
    primaryButton: {
        backgroundColor: '#00ffcc',
    },
    actionText: {
        color: '#fff',
        marginLeft: 8,
        fontWeight: '700',
        fontSize: 16,
    },
});
