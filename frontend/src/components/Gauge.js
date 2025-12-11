import React from 'react';
import { Box } from '@mui/material';

export function Gauge({ value, color = '#667eea' }) {
  const percentage = Math.min(Math.max(value, 0), 100);
  const angle = (percentage / 100) * 180 - 90;
  const radius = 60;
  const circumference = Math.PI * radius;

  return (
    <Box
      sx={{
        position: 'relative',
        width: 140,
        height: 80,
        margin: '0 auto'
      }}
    >
      <svg
        width="140"
        height="80"
        viewBox="0 0 140 80"
        style={{ overflow: 'visible' }}
      >
        {/* Background arc */}
        <path
          d={`M 20 70 A ${radius} ${radius} 0 0 1 120 70`}
          fill="none"
          stroke="#e0e0e0"
          strokeWidth="12"
          strokeLinecap="round"
        />
        {/* Value arc */}
        <path
          d={`M 20 70 A ${radius} ${radius} 0 ${percentage > 50 ? 1 : 0} 1 ${
            70 + radius * Math.cos((angle * Math.PI) / 180)
          } ${70 + radius * Math.sin((angle * Math.PI) / 180)}`}
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeLinecap="round"
          style={{
            transition: 'all 0.5s ease'
          }}
        />
      </svg>
    </Box>
  );
}
