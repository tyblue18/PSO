import React from 'react';
import {
  Paper,
  Typography,
  Box,
  Grid,
  Card,
  CardContent
} from '@mui/material';
import {
  Science as ScienceIcon,
  Speed as SpeedIcon,
  Assessment as AssessmentIcon,
  Code as CodeIcon
} from '@mui/icons-material';

function About() {
  const features = [
    {
      icon: <ScienceIcon sx={{ fontSize: 40, color: '#667eea' }} />,
      title: 'PSO Algorithm',
      description: 'Particle Swarm Optimization for adaptive threshold selection'
    },
    {
      icon: <SpeedIcon sx={{ fontSize: 40, color: '#764ba2' }} />,
      title: 'High Performance',
      description: 'Optimized with vectorized operations and parallel processing'
    },
    {
      icon: <AssessmentIcon sx={{ fontSize: 40, color: '#f093fb' }} />,
      title: 'Comprehensive Metrics',
      description: 'Dice, IoU, Precision, and Recall with detailed statistics'
    },
    {
      icon: <CodeIcon sx={{ fontSize: 40, color: '#4facfe' }} />,
      title: 'Modern Stack',
      description: 'Built with React, Flask, and advanced ML techniques'
    }
  ];

  return (
    <Box>
      <Paper sx={{ p: 4, mb: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 600 }}>
          About This Application
        </Typography>
        <Typography variant="body1" paragraph>
          This interactive web application demonstrates Particle Swarm Optimization (PSO)
          for automated brain tumor segmentation on medical MRI images. The system uses
          advanced optimization algorithms to find optimal thresholds for binary
          segmentation, achieving robust performance through adaptive optimization.
        </Typography>
      </Paper>

      <Grid container spacing={3} sx={{ mb: 3 }}>
        {features.map((feature, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card sx={{ height: '100%', textAlign: 'center' }}>
              <CardContent>
                <Box sx={{ mb: 2 }}>{feature.icon}</Box>
                <Typography variant="h6" gutterBottom>
                  {feature.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {feature.description}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Technical Details
        </Typography>
        <Typography variant="body2" component="div">
          <strong>Algorithm:</strong> Global-best PSO with 30 particles, 40 iterations
          <br />
          <strong>Metrics:</strong> Dice Coefficient, IoU, Precision, Recall
          <br />
          <strong>Dataset:</strong> BraTS 2017/2018 (154K+ medical images)
          <br />
          <strong>Performance:</strong> 2-4× faster with parallel processing
        </Typography>
      </Paper>
    </Box>
  );
}

export default About;
