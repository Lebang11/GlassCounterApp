import { StatusBar } from 'expo-status-bar';
import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, SafeAreaView, ActivityIndicator } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native'; // registers react-native backend

import CameraScreen from './components/CameraScreen';
import ResultScreen from './components/ResultScreen';

export default function App() {
  const [isTfReady, setIsTfReady] = useState(false);
  const [photo, setPhoto] = useState(null);

  useEffect(() => {
    async function prepare() {
      await tf.ready(); // Initialize TensorFlow.js dummy backend connection
      setIsTfReady(true);
    }
    prepare();
  }, []);

  if (!isTfReady) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#00ffcc" />
        <Text style={styles.loadingText}>Initializing ML Engine...</Text>
        <StatusBar style="light" />
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {photo ? (
        <ResultScreen photo={photo} onRetake={() => setPhoto(null)} />
      ) : (
        <CameraScreen onPictureTaken={(p) => setPhoto(p)} />
      )}
      <StatusBar style="light" />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  loadingContainer: {
    flex: 1,
    backgroundColor: '#000',
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: {
    color: '#00ffcc',
    marginTop: 20,
    fontSize: 18,
    fontWeight: 'bold',
  }
});
