import React, { useState } from 'react';
import {
  Paper,
  Box,
  Button,
  Typography,
  Alert
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import { Upload as UploadIcon } from '@mui/icons-material';

function ImageUploader({ onProcess, disabled }) {
  const [imageFile, setImageFile] = useState(null);
  const [maskFile, setMaskFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [maskPreview, setMaskPreview] = useState(null);

  const imageDropzone = useDropzone({
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setImageFile(acceptedFiles[0]);
        const reader = new FileReader();
        reader.onload = () => setImagePreview(reader.result);
        reader.readAsDataURL(acceptedFiles[0]);
      }
    },
    disabled
  });

  const maskDropzone = useDropzone({
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setMaskFile(acceptedFiles[0]);
        const reader = new FileReader();
        reader.onload = () => setMaskPreview(reader.result);
        reader.readAsDataURL(acceptedFiles[0]);
      }
    },
    disabled
  });

  const handleProcess = () => {
    if (imageFile) {
      onProcess(imageFile, maskFile);
    }
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Upload Images
      </Typography>

      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" gutterBottom>
          MRI Image (Required)
        </Typography>
        <Box
          {...imageDropzone.getRootProps()}
          sx={{
            border: '2px dashed #ccc',
            borderRadius: 2,
            p: 3,
            textAlign: 'center',
            cursor: 'pointer',
            '&:hover': {
              borderColor: '#667eea',
              backgroundColor: '#f5f5f5'
            }
          }}
        >
          <input {...imageDropzone.getInputProps()} />
          <UploadIcon sx={{ fontSize: 48, color: '#999', mb: 1 }} />
          <Typography>
            {imageFile ? imageFile.name : 'Drop image here or click to upload'}
          </Typography>
        </Box>
        {imagePreview && (
          <Box sx={{ mt: 2 }}>
            <img
              src={imagePreview}
              alt="Preview"
              style={{ maxWidth: '100%', borderRadius: 8 }}
            />
          </Box>
        )}
      </Box>

      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" gutterBottom>
          Ground Truth Mask (Optional)
        </Typography>
        <Box
          {...maskDropzone.getRootProps()}
          sx={{
            border: '2px dashed #ccc',
            borderRadius: 2,
            p: 3,
            textAlign: 'center',
            cursor: 'pointer',
            '&:hover': {
              borderColor: '#667eea',
              backgroundColor: '#f5f5f5'
            }
          }}
        >
          <input {...maskDropzone.getInputProps()} />
          <UploadIcon sx={{ fontSize: 48, color: '#999', mb: 1 }} />
          <Typography>
            {maskFile ? maskFile.name : 'Drop mask here or click to upload'}
          </Typography>
        </Box>
        {maskPreview && (
          <Box sx={{ mt: 2 }}>
            <img
              src={maskPreview}
              alt="Mask Preview"
              style={{ maxWidth: '100%', borderRadius: 8 }}
            />
          </Box>
        )}
      </Box>

      <Button
        variant="contained"
        fullWidth
        size="large"
        onClick={handleProcess}
        disabled={!imageFile || disabled}
        sx={{
          background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
          '&:hover': {
            background: 'linear-gradient(90deg, #5568d3 0%, #653a91 100%)'
          }
        }}
      >
        Process Image
      </Button>

      {!maskFile && (
        <Alert severity="info" sx={{ mt: 2 }}>
          Upload a mask to see segmentation results and metrics.
        </Alert>
      )}
    </Paper>
  );
}

export default ImageUploader;
