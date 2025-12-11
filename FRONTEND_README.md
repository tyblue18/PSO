# Interactive Frontend for PSO Brain Tumor Segmentation

Professional web application with interactive visualizations to impress recruiters and showcase your technical skills.

## 🎯 Features

- **Interactive UI**: Modern, professional design with Material-UI
- **Real-time Processing**: Upload images and see results instantly
- **Advanced Visualizations**: Gauge charts, progress bars, side-by-side comparisons
- **Comprehensive Metrics**: Dice, IoU, Precision, Recall with detailed statistics
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Two Options**: Streamlit (quick) and React+Flask (impressive)

## 🚀 Quick Start

### Option 1: Streamlit App (Easiest)

```bash
# Install dependencies
pip install streamlit plotly

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Option 2: React + Flask (Most Impressive)

#### Backend (Flask API)

```bash
# Install Flask dependencies
pip install flask flask-cors

# Run the API
python api.py
```

API runs at `http://localhost:5000`

#### Frontend (React)

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

Frontend runs at `http://localhost:3000`

## 📁 Project Structure

```
.
├── app.py                 # Streamlit web application
├── api.py                 # Flask REST API
├── frontend/              # React frontend
│   ├── src/
│   │   ├── App.js        # Main app component
│   │   └── components/   # React components
│   │       ├── ImageUploader.js
│   │       ├── MetricsDisplay.js
│   │       ├── Visualization.js
│   │       └── ...
│   └── package.json
└── FRONTEND_README.md    # This file
```

## 🎨 Features Showcase

### Streamlit App
- ✅ Drag-and-drop image upload
- ✅ Real-time PSO processing
- ✅ Interactive metric gauges
- ✅ Side-by-side visualizations
- ✅ Professional UI with custom CSS
- ✅ Configuration sidebar
- ✅ Batch processing tab

### React + Flask App
- ✅ Modern Material-UI design
- ✅ RESTful API architecture
- ✅ Advanced gauge visualizations
- ✅ Responsive grid layout
- ✅ Error handling and loading states
- ✅ Professional color scheme
- ✅ Mobile-friendly interface

## 🚢 Deployment

### Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy!

### Heroku (React + Flask)

```bash
# Create Procfile
echo "web: python api.py" > Procfile
echo "web: npm start" >> Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

### Docker

```dockerfile
# Dockerfile example
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 💡 Why This Impresses Recruiters

1. **Full-Stack Skills**: Shows you can build complete applications
2. **Modern Tech Stack**: React, Flask, Streamlit demonstrate versatility
3. **Professional UI**: Material-UI shows attention to design
4. **Real-World Application**: Medical AI is highly relevant
5. **Performance**: Optimized code with parallel processing
6. **Documentation**: Well-documented code and README
7. **Deployment Ready**: Can be deployed to production

## 📊 Screenshots

The application includes:
- Gradient header with professional typography
- Interactive metric gauges
- Side-by-side image comparisons
- Real-time processing feedback
- Responsive design

## 🔧 Customization

### Change Colors
Edit the gradient colors in `app.py` or `App.js`:
```python
# Streamlit
background: linear-gradient(90deg, #667eea 0%, #764ba2 100%)
```

### Add Features
- Batch processing visualization
- Historical results tracking
- Export functionality
- User authentication
- Database integration

## 📝 Notes

- The Streamlit app is faster to set up and great for demos
- The React+Flask app shows more technical depth
- Both can be deployed for free on various platforms
- Consider adding authentication for production use

## 🎓 Learning Resources

- [Streamlit Docs](https://docs.streamlit.io/)
- [React Docs](https://react.dev/)
- [Flask Docs](https://flask.palletsprojects.com/)
- [Material-UI Docs](https://mui.com/)

## 🤝 Contributing

Feel free to enhance the frontend with:
- More visualizations
- Better error handling
- Performance optimizations
- Additional features
