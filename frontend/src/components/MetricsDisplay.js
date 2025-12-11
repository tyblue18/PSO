import React from 'react';
import {
  Paper,
  Grid,
  Typography,
  Box,
  LinearProgress
} from '@mui/material';
import { Gauge } from './Gauge';

function MetricsDisplay({ metrics, threshold }) {
  if (!metrics) {
    return (
      <Paper sx={{ p: 3 }}>
        <Typography>Upload a mask to see metrics</Typography>
      </Paper>
    );
  }

  const metricData = [
    { label: 'Dice Coefficient', value: metrics.dice, color: '#667eea' },
    { label: 'IoU', value: metrics.iou, color: '#764ba2' },
    { label: 'Precision', value: metrics.precision, color: '#f093fb' },
    { label: 'Recall', value: metrics.recall, color: '#4facfe' }
  ];

  return (
    <Paper sx={{ p: 3, mb: 3 }}>
      <Typography variant="h6" gutterBottom>
        Evaluation Metrics
      </Typography>

      {threshold && (
        <Box sx={{ mb: 3, p: 2, bgcolor: '#f5f5f5', borderRadius: 2 }}>
          <Typography variant="subtitle2" color="text.secondary">
            Optimal Threshold
          </Typography>
          <Typography variant="h5" sx={{ fontWeight: 600 }}>
            {threshold.toFixed(4)}
          </Typography>
        </Box>
      )}

      <Grid container spacing={2}>
        {metricData.map((metric, index) => (
          <Grid item xs={6} key={index}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                {metric.label}
              </Typography>
              <Gauge value={metric.value * 100} color={metric.color} />
              <Typography variant="h6" sx={{ mt: 1 }}>
                {(metric.value * 100).toFixed(2)}%
              </Typography>
            </Box>
          </Grid>
        ))}
      </Grid>

      {/* Progress bars */}
      <Box sx={{ mt: 3 }}>
        {metricData.map((metric, index) => (
          <Box key={index} sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
              <Typography variant="body2">{metric.label}</Typography>
              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                {(metric.value * 100).toFixed(2)}%
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={metric.value * 100}
              sx={{
                height: 8,
                borderRadius: 4,
                backgroundColor: '#e0e0e0',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: metric.color,
                  borderRadius: 4
                }
              }}
            />
          </Box>
        ))}
      </Box>
    </Paper>
  );
}

export default MetricsDisplay;
