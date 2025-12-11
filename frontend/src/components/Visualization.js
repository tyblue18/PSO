import React from 'react';
import {
  Paper,
  Grid,
  Typography,
  Box
} from '@mui/material';

function Visualization({ original, mask, prediction }) {
  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Segmentation Results
      </Typography>
      <Grid container spacing={2}>
        <Grid item xs={4}>
          <Box>
            <Typography variant="caption" display="block" gutterBottom>
              Input Image
            </Typography>
            <img
              src={original}
              alt="Original"
              style={{
                width: '100%',
                borderRadius: 8,
                border: '2px solid #e0e0e0'
              }}
            />
          </Box>
        </Grid>
        <Grid item xs={4}>
          <Box>
            <Typography variant="caption" display="block" gutterBottom>
              Ground Truth
            </Typography>
            <Box sx={{ position: 'relative' }}>
              <img
                src={original}
                alt="Original"
                style={{
                  width: '100%',
                  borderRadius: 8,
                  border: '2px solid #e0e0e0'
                }}
              />
              <img
                src={mask}
                alt="Mask"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  borderRadius: 8,
                  opacity: 0.6,
                  mixBlendMode: 'multiply'
                }}
              />
            </Box>
          </Box>
        </Grid>
        <Grid item xs={4}>
          <Box>
            <Typography variant="caption" display="block" gutterBottom>
              PSO Prediction
            </Typography>
            <Box sx={{ position: 'relative' }}>
              <img
                src={original}
                alt="Original"
                style={{
                  width: '100%',
                  borderRadius: 8,
                  border: '2px solid #e0e0e0'
                }}
              />
              <img
                src={prediction}
                alt="Prediction"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  borderRadius: 8,
                  opacity: 0.6,
                  mixBlendMode: 'multiply'
                }}
              />
            </Box>
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );
}

export default Visualization;
