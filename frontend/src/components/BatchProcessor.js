import React from 'react';
import { Paper, Typography, Box } from '@mui/material';

function BatchProcessor() {
  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Batch Processing
      </Typography>
      <Typography color="text.secondary">
        Batch processing feature coming soon! This will allow you to upload
        multiple images and process them in parallel, generating comprehensive
        statistics and visualizations.
      </Typography>
    </Paper>
  );
}

export default BatchProcessor;
