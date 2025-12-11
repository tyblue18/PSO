import React, { useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Grid,
  Paper,
  Button,
  CircularProgress,
  Alert,
  Tabs,
  Tab,
  Card,
  CardContent,
  CardMedia
} from '@mui/material';
import {
  Upload as UploadIcon,
  Science as ScienceIcon,
  Analytics as AnalyticsIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import axios from 'axios';
import ImageUploader from './components/ImageUploader';
import MetricsDisplay from './components/MetricsDisplay';
import Visualization from './components/Visualization';
import BatchProcessor from './components/BatchProcessor';
import About from './components/About';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

function App() {
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleProcessImage = async (imageFile, maskFile) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      if (maskFile) {
        formData.append('mask', maskFile);
      }

      // Convert to base64
      const imageBase64 = await fileToBase64(imageFile);
      const maskBase64 = maskFile ? await fileToBase64(maskFile) : null;

      const response = await axios.post(`${API_URL}/process`, {
        image: imageBase64,
        mask: maskBase64
      });

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = (error) => reject(error);
    });
  };

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ textAlign: 'center', mb: 4 }}>
        <Typography
          variant="h2"
          component="h1"
          sx={{
            fontWeight: 700,
            background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            mb: 1
          }}
        >
          🧠 PSO Brain Tumor Segmentation
        </Typography>
        <Typography variant="h6" color="text.secondary">
          Advanced Medical Image Segmentation using Particle Swarm Optimization
        </Typography>
      </Box>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={tabValue}
          onChange={(e, newValue) => setTabValue(newValue)}
          variant="fullWidth"
          indicatorColor="primary"
          textColor="primary"
        >
          <Tab icon={<UploadIcon />} label="Process Image" />
          <Tab icon={<AnalyticsIcon />} label="Batch Analysis" />
          <Tab icon={<InfoIcon />} label="About" />
        </Tabs>
      </Paper>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Loading Indicator */}
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress size={60} />
        </Box>
      )}

      {/* Tab Content */}
      {tabValue === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <ImageUploader
              onProcess={handleProcessImage}
              disabled={loading}
            />
          </Grid>
          <Grid item xs={12} md={6}>
            {result && (
              <>
                <MetricsDisplay metrics={result.metrics} threshold={result.threshold} />
                {result.has_mask && (
                  <Visualization
                    original={result.processed_image}
                    mask={result.mask}
                    prediction={result.prediction}
                  />
                )}
              </>
            )}
          </Grid>
        </Grid>
      )}

      {tabValue === 1 && <BatchProcessor />}
      {tabValue === 2 && <About />}
    </Container>
  );
}

export default App;
