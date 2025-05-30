# 🚁 Drone Detection System

A modern web application that uses AI-powered YOLO (You Only Look Once) object detection to identify and highlight drones in images and videos. The system supports both file uploads and YouTube video processing.

## ✨ Features

- **🖼️ Image Detection**: Upload JPG, PNG, or WEBP images for drone detection
- **🎥 Video Processing**: Process MP4 videos with frame-by-frame drone detection
- **🔗 YouTube Integration**: Paste YouTube video URLs (including Shorts) for analysis
- **🎯 Real-time Processing**: Advanced YOLO model for accurate drone identification
- **💾 Memory-Efficient**: No permanent file storage - everything processed in memory
- **📱 Responsive Design**: Works seamlessly on desktop and mobile devices
- **⚡ Fast Processing**: Optimized for quick results with visual feedback

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **Ultralytics YOLO** - AI object detection model
- **OpenCV** - Computer vision processing
- **yt-dlp** - YouTube video downloading
- **Flask-CORS** - Cross-origin resource sharing

### Frontend
- **React** - User interface framework (bootstrapped with Create React App)
- **Modern JavaScript (ES6+)**
- **CSS3** - Responsive styling
- **HTML5** - Semantic markup

## 📋 Prerequisites

- Python 3.8 or higher
- Node.js 14.0 or higher
- npm or yarn package manager

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/drone-detector.git
cd drone-detector
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# or with yarn
yarn install
```

### 4. Model Setup

Place your trained YOLO model file (`best.pt`) in the `backend/models/` directory.

## 🏃‍♂️ Running the Application

### Start the Backend Server

```bash
cd backend
python app.py
```

The backend server will start on `http://localhost:5001`

### Start the Frontend Development Server

```bash
cd frontend
npm start
```

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

## 📁 Project Structure

```
drone-detector/
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── yolo_utils.py         # YOLO processing utilities
│   ├── models/
│   │   └── best.pt           # YOLO model file
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUpload.js # Main upload component
│   │   │   └── FileUpload.css # Component styles
│   │   ├── App.js            # Main React app
│   │   └── index.js          # App entry point
│   ├── public/
│   └── package.json          # Node.js dependencies
└── README.md
```

## 🎯 Usage

### File Upload
1. Click "Choose File" or drag and drop an image/video
2. Supported formats: JPG, PNG, WEBP (images), MP4 (videos)
3. Maximum file size: 100MB
4. Click "📤 Analyze File" to process

### YouTube Processing
1. Paste a YouTube URL in the input field
2. Supported formats:
   - `https://www.youtube.com/watch?v=VIDEO_ID`
   - `https://youtu.be/VIDEO_ID`
   - `https://www.youtube.com/shorts/VIDEO_ID`
3. Maximum duration: 10 minutes
4. Click "🔗 Process YouTube Video"

### Download Results
- View original and processed media side by side
- Download both original and detection results
- Results include bounding boxes around detected drones

## ⚙️ API Endpoints

### File Upload
```
POST /api/upload
Content-Type: multipart/form-data
Body: file (image/video)
```

### YouTube Processing
```
POST /api/youtube
Content-Type: application/json
Body: {"url": "youtube_video_url"}
```

### Health Check
```
GET /test
Response: {"message": "Test endpoint working!", "status": "ok"}
```

## 🔧 Configuration

### Backend Configuration (app.py)
- **Host**: `localhost`
- **Port**: `5001`
- **Debug**: `False` (production mode)
- **File size limit**: 100MB
- **Processing timeout**: 10 minutes

### Frontend Configuration
- **Development server**: `localhost:3000`
- **API base URL**: `http://localhost:5001`

## 📊 Performance

- **Image processing**: ~2-5 seconds
- **Video processing**: ~1-2 minutes per minute of video
- **YouTube download**: ~30 seconds - 2 minutes (depending on video length)
- **Memory usage**: Optimized for minimal disk storage
- **Supported resolutions**: Up to 1080p (720
