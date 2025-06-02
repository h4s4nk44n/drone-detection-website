// src/FileUpload.js
import React, { useState, useEffect, useRef } from 'react';

const FileUpload = () => {
    const [files, setFiles] = useState([]);
    const [results, setResults] = useState([]); // Array of {original, processed, filename, id}
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState('');
    const [serverStatus, setServerStatus] = useState('checking');
    const fileInputRef = useRef(null);
    const [error, setError] = useState('');
    const [dragActive, setDragActive] = useState(false);
    const [processingProgress, setProcessingProgress] = useState(0);
    const [currentlyProcessing, setCurrentlyProcessing] = useState('');
    
    // Training data extraction state
    const [extractingTrainingData, setExtractingTrainingData] = useState(false);
    const [showTrainingExtraction, setShowTrainingExtraction] = useState(false);
    const [confidenceThreshold, setConfidenceThreshold] = useState(0.3);
    const [bulkTrainingData, setBulkTrainingData] = useState(null); // For storing the combined bulk training data

    // Camera functionality state
    const [showCamera, setShowCamera] = useState(false);
    const [cameraStream, setCameraStream] = useState(null);
    const [isRecording, setIsRecording] = useState(false);
    const [mediaRecorder, setMediaRecorder] = useState(null);
    const [cameraError, setCameraError] = useState('');
    const [recordingTime, setRecordingTime] = useState(0);
    const [recordingTimer, setRecordingTimer] = useState(null);
    const cameraVideoRef = useRef(null);
    const canvasRef = useRef(null);
    const [isCameraViewfinderReady, setIsCameraViewfinderReady] = useState(false);
    const [availableCameras, setAvailableCameras] = useState([]);
    const [selectedCameraId, setSelectedCameraId] = useState('');

    // Constants
    const MAX_RECORDING_TIME = 120; // 120 seconds = 2 minutes

    // Test server connection on component mount
    useEffect(() => {
        const testServer = async () => {
            try {
                console.log('🔄 Testing server connection...');
                const response = await fetch('https://drone-detection-686868741947.europe-west1.run.app/test', {
                    method: 'GET',
                    mode: 'cors',
                    headers: { 'Content-Type': 'application/json' }
                });
                if (response.ok) {
                    setServerStatus('connected');
                } else {
                    setServerStatus('error');
                }
            } catch (error) {
                console.error('❌ Server test error:', error);
                setServerStatus('error');
            }
        };
        testServer();
        const interval = setInterval(testServer, 30000);
        return () => clearInterval(interval);
    }, []);

    // Cleanup camera stream tracks when cameraStream changes or on component unmount
    useEffect(() => {
        const currentStreamToClean = cameraStream;
        return () => {
            if (currentStreamToClean) {
                console.log('🧹 useEffect[cameraStream] cleanup: Stopping tracks for stream instance:', currentStreamToClean.id);
                currentStreamToClean.getTracks().forEach(track => {
                    if (track.readyState === 'live') track.stop();
                });
            }
        };
    }, [cameraStream]);

    // Separate effect for cleaning up the recordingTimer on component unmount
    useEffect(() => {
        const timerIdToClearOnUnmount = recordingTimer;
        return () => {
            if (timerIdToClearOnUnmount) {
                console.log('🧹 useEffect[] unmount (safety net): Clearing recordingTimer ID:', timerIdToClearOnUnmount);
                clearInterval(timerIdToClearOnUnmount);
            }
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); // Empty dependency: runs only on mount and unmount

    // Effect to setup video element when cameraStream is ready and showCamera is true
    useEffect(() => {
        if (showCamera && cameraStream && cameraVideoRef.current) {
            const video = cameraVideoRef.current;
            console.log('📹 useEffect [camera] active: Configuring video element.');
            video.muted = true;
            video.srcObject = cameraStream;
            setIsCameraViewfinderReady(false);

            let resolved = false;
            const timeoutDuration = 15000;

            const cleanupEventListeners = () => {
                video.removeEventListener('loadedmetadata', onLoadedMetadata);
                video.removeEventListener('canplay', onCanPlay);
                video.removeEventListener('playing', onPlaying);
                video.removeEventListener('stalled', onStalled);
                video.removeEventListener('error', onErrorEvent); // Renamed to avoid conflict
            };

            const succeed = (eventName) => {
                if (resolved) return;
                resolved = true;
                cleanupEventListeners();
                console.log(`✅ Video event in useEffect [camera]: ${eventName}. Viewfinder ready.`);
                setIsCameraViewfinderReady(true);
            };

            const fail = (errorMsg, errorObj) => {
                if (resolved) return;
                resolved = true;
                cleanupEventListeners();
                console.error(`❌ ${errorMsg} in useEffect [camera]`, errorObj || '');
                setCameraError(`Camera stream error: ${errorMsg.split('.')[0]}. Try restarting camera.`);
                setIsCameraViewfinderReady(false);
            };

            const onLoadedMetadata = () => console.log('📹 Event [camera]: loadedmetadata. Dimensions:', video.videoWidth, 'x', video.videoHeight);
            const onCanPlay = () => succeed('canplay');
            const onPlaying = () => succeed('playing');
            const onStalled = () => console.warn('📹 Event [camera]: stalled');
            const onErrorEvent = (event) => { // Renamed to avoid conflict
                let errorDetail = 'unknown error';
                if (event?.target?.error) {
                    const vidError = event.target.error;
                    errorDetail = `code ${vidError.code}, ${vidError.message}`;
                }
                fail(`Video element stream error (${errorDetail})`, event);
            };

            video.addEventListener('loadedmetadata', onLoadedMetadata);
            video.addEventListener('canplay', onCanPlay);
            video.addEventListener('playing', onPlaying);
            video.addEventListener('stalled', onStalled);
            video.addEventListener('error', onErrorEvent);
            
            video.play().then(() => {
                console.log('✅ video.play() promise resolved in useEffect [camera].');
                setTimeout(() => {
                    if (!resolved && video.readyState >= 3) {
                        succeed('play_promise_and_readyState_OK');
                    }
                }, 1000);
            }).catch(playError => {
                console.warn('⚠️ video.play() promise rejected:', playError.name, playError.message);
            });

            const timer = setTimeout(() => {
                if (!resolved) fail(`Video setup timeout [camera] after ${timeoutDuration / 1000}s. ReadyState: ${video.readyState}`);
            }, timeoutDuration);

            return () => {
                console.log('🧹 useEffect cleanup [camera]: Releasing srcObject and listeners.');
                clearTimeout(timer);
                cleanupEventListeners();
                if (video && video.srcObject) video.srcObject = null;
                setIsCameraViewfinderReady(false);
            };
        } else if (!cameraStream && showCamera) {
            setIsCameraViewfinderReady(false);
        }
    }, [cameraStream, showCamera]);


    const handleFileSelection = (selectedFiles) => {
        const validImageTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
        const validVideoTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/webm']; // Added webm
        const validTypes = [...validImageTypes, ...validVideoTypes];
        
        const ProcessedFiles = selectedFiles.filter(file => validTypes.includes(file.type));
        
        if (ProcessedFiles.length === 0 && selectedFiles.length > 0) {
            setError('Please select valid files (Images: JPG, PNG, WEBP | Videos: MP4, AVI, MOV, WEBM)');
            return;
        }
        if (selectedFiles.length !== ProcessedFiles.length) {
            setError(`${selectedFiles.length - ProcessedFiles.length} files were skipped (unsupported types).`);
        }

        const imageFiles = ProcessedFiles.filter(f => validImageTypes.includes(f.type));
        const videoFiles = ProcessedFiles.filter(f => validVideoTypes.includes(f.type));

        if (videoFiles.length > 1) {
            setError('Only one video can be processed at a time. Please select one video and unlimited images.');
            return;
        }

        const totalSize = ProcessedFiles.reduce((sum, file) => sum + file.size, 0);
        const totalSizeMB = totalSize / (1024 * 1024);
        const maxTotalSizeMB = 200; 

        if (totalSizeMB > maxTotalSizeMB) {
            setError(`Total file size too large (${totalSizeMB.toFixed(1)}MB). Maximum is ${maxTotalSizeMB}MB.`);
            return;
        }

        const oversizedFiles = ProcessedFiles.filter(file => {
            const fileSizeMB = file.size / (1024 * 1024);
            const isVideo = validVideoTypes.includes(file.type);
            const maxSize = isVideo ? 75 : 25;
            return fileSizeMB > maxSize;
        });
        
        if (oversizedFiles.length > 0) {
            let errorMsg = 'Some files are too large: ';
            if (oversizedFiles.some(f => validVideoTypes.includes(f.type))) errorMsg += `Videos must be ≤75MB. `;
            if (oversizedFiles.some(f => validImageTypes.includes(f.type))) errorMsg += `Images must be ≤25MB.`;
            setError(errorMsg.trim());
            return;
        }

        setFiles(ProcessedFiles); // Set the filtered valid files
        setError(''); // Clear previous errors
        
        let selectionMessage = `✅ Selected ${imageFiles.length} image(s) and ${videoFiles.length} video(s) (${totalSizeMB.toFixed(1)}MB total).`;
        if (imageFiles.length > 0 && videoFiles.length === 0) selectionMessage = `✅ Selected ${imageFiles.length} image(s) (${totalSizeMB.toFixed(1)}MB total).`;
        if (imageFiles.length === 0 && videoFiles.length > 0) selectionMessage = `✅ Selected ${videoFiles.length} video(s) (${totalSizeMB.toFixed(1)}MB total).`;
        if (ProcessedFiles.length === 0) selectionMessage = '';

        console.log(selectionMessage || "No valid files selected or selection cleared.");
        // setMessage(selectionMessage); // Optional: display this message in the UI
    };

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelection(Array.from(e.dataTransfer.files));
        }
    };

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelection(Array.from(e.target.files));
            e.target.value = null; // Reset file input to allow selecting the same file again
        }
    };

    const uploadLargeFile = async (file) => { // Assuming this function is defined elsewhere or you will add it
        const CHUNK_SIZE = 25 * 1024 * 1024; 
        const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
        const uploadId = Date.now().toString();
        
        console.log(`📦 Splitting ${(file.size / (1024 * 1024)).toFixed(1)}MB file into ${totalChunks} chunks`);
        
        for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
            const start = chunkIndex * CHUNK_SIZE;
            const end = Math.min(start + CHUNK_SIZE, file.size);
            const chunk = file.slice(start, end);
            
            const chunkFormData = new FormData();
            chunkFormData.append('chunk', chunk);
            chunkFormData.append('chunkIndex', chunkIndex.toString());
            chunkFormData.append('totalChunks', totalChunks.toString());
            chunkFormData.append('uploadId', uploadId);
            chunkFormData.append('fileName', file.name);
            
            setMessage(`🔄 Uploading chunk ${chunkIndex + 1}/${totalChunks} for ${file.name}...`);
            
            const response = await fetch('https://drone-detection-686868741947.europe-west1.run.app/api/upload-chunk', {
                method: 'POST',
                body: chunkFormData
            });
            
            if (!response.ok) throw new Error(`Chunk ${chunkIndex + 1} upload failed`);
        }
        
        setMessage(`🤖 Processing complete file ${file.name}... This may take minutes...`);
        const processResponse = await fetch('https://drone-detection-686868741947.europe-west1.run.app/api/process-uploaded', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ uploadId, fileName: file.name }),
        });
        return { processResponse }; // Removed controller as it wasn't used for abort
    };

    const downloadFileInChunks = async (fileInfo, fileType) => { // Assuming this function is defined elsewhere or you will add it
        const chunks = [];
        console.log(`📥 Starting chunked download: ${fileInfo.chunks} chunks (${(fileInfo.size / (1024*1024)).toFixed(2)}MB) for ${fileType}`);
        await new Promise(resolve => setTimeout(resolve, 1000)); // Initial delay before download starts

            for (let i = 0; i < fileInfo.chunks; i++) {
                const chunkUrl = `https://drone-detection-686868741947.europe-west1.run.app/api/download-chunk/${fileInfo.file_id}/${i}`;
                let retryCount = 0;
                const maxRetries = 3;
                let chunkSuccess = false;
                
                while (!chunkSuccess && retryCount < maxRetries) {
                    try {
                    if (retryCount > 0) {
                        const currentDelayRetryCount = retryCount; // Capture retryCount for this iteration's delay
                        await new Promise(resolve => setTimeout(resolve, Math.min(1000 * Math.pow(2, currentDelayRetryCount), 8000)));
                    }
                    
                    const response = await fetch(chunkUrl, { method: 'GET', mode: 'cors', headers: { 'Accept': 'application/json' }});
                        if (!response.ok) {
                        const errorData = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
                        throw new Error(`Chunk ${i+1} download failed: ${errorData.error}`);
                    }
                        const chunkData = await response.json();
                    if (!chunkData.chunk_data) throw new Error(`Chunk ${i+1} has no data`);
                    
                            const binaryString = atob(chunkData.chunk_data);
                            const bytes = new Uint8Array(binaryString.length);
                    for (let j = 0; j < binaryString.length; j++) bytes[j] = binaryString.charCodeAt(j);
                            chunks.push(bytes);
                    chunkSuccess = true;
                    } catch (fetchError) {
                        retryCount++;
                    if (retryCount >= maxRetries) throw new Error(`Failed chunk ${i+1} after ${maxRetries} attempts: ${fetchError.message}`);
                }
            }
            if (i < fileInfo.chunks - 1) await new Promise(resolve => setTimeout(resolve, 200)); // Small delay between chunks
        }
        
        const combined = new Uint8Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
            let offset = 0;
            for (const chunk of chunks) {
                combined.set(chunk, offset);
                offset += chunk.length;
            }
            
            const blob = new Blob([combined], { type: fileInfo.mime_type });
            const objectUrl = URL.createObjectURL(blob);
            
        try { // Cleanup server file
            await fetch(`https://drone-detection-686868741947.europe-west1.run.app/api/cleanup-download/${fileInfo.file_id}`, { method: 'POST', mode: 'cors' });
        } catch (cleanupError) { console.warn('⚠️ Failed to cleanup server file:', cleanupError); }
        
        return objectUrl;
    };
    
    const processAllFiles = async () => {
        if (files.length === 0) {
            alert('Please select files first');
            return;
        }

        setLoading(true);
        setResults([]);
        setProcessingProgress(0);
        setMessage(`🔄 Processing ${files.length} files...`);
        setError(''); 

        const newResultsList = []; // Use a local list to build results

        try {
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                setCurrentlyProcessing(file.name);
                setProcessingProgress(((i) / files.length) * 100);

                const fileSizeMB = file.size / (1024 * 1024);
                const isVideo = file.type.startsWith('video/');
                
                console.log(`🔄 Processing file ${i + 1}/${files.length}: ${file.name} (${fileSizeMB.toFixed(1)}MB, ${isVideo ? 'video' : 'image'})`);

                try {
                    let response;
                    if (isVideo && fileSizeMB > 25) {
                        console.log('🔄 Using chunked upload for large video...');
                        const { processResponse } = await uploadLargeFile(file);
                        response = processResponse;
                    } else {
                        const formData = new FormData();
                        formData.append('file', file);
                        response = await fetch('https://drone-detection-686868741947.europe-west1.run.app/api/upload', {
                            method: 'POST',
                            body: formData
                        });
                    }

                    if (!response.ok) {
                        const errorData = await response.json().catch(() => ({error: `Server error: ${response.status}`}));
                        throw new Error(errorData.error || `HTTP ${response.status}`);
                    }

                    const data = await response.json();
                    let originalUrl, processedUrl;
                    
                    if (data.chunked_download) {
                        console.log(`📥 Downloading chunked result for ${file.name}...`);
                        originalUrl = await downloadFileInChunks(data.original_file, 'original');
                        processedUrl = await downloadFileInChunks(data.processed_file, 'processed');
                    } else {
                        originalUrl = data.original_file || data.original; // Ensure fallback if structure varies
                        processedUrl = data.output_file || data.processed;
                    }

                    newResultsList.push({
                        id: Date.now() + i + Math.random(), // Add random to ensure uniqueness for fast processing
                        filename: file.name,
                        original: originalUrl,
                        processed: processedUrl,
                        fileSize: fileSizeMB.toFixed(2),
                        fileType: isVideo ? 'video' : 'image'
                    });
                    console.log(`✅ Completed ${file.name}`);
                } catch (fileError) {
                    console.error(`❌ Error processing ${file.name}:`, fileError);
                    newResultsList.push({
                        id: Date.now() + i + Math.random(),
                        filename: file.name,
                        error: fileError.message,
                        fileSize: fileSizeMB.toFixed(2),
                        fileType: isVideo ? 'video' : 'image'
                    });
                }
                setResults([...newResultsList]); // Update results state after each file
            }

            setProcessingProgress(100);
            const successCount = newResultsList.filter(r => !r.error).length;
            const failCount = newResultsList.filter(r => r.error).length;
            
            setMessage(`✅ Batch complete: ${successCount} successful, ${failCount} failed.`);

        } catch (batchError) { // This catch might be redundant if individual file errors are handled well
            console.error('❌ Outer batch processing failed:', batchError);
            setMessage(`❌ Batch processing error: ${batchError.message}`);
            setError(`Batch error: ${batchError.message}`);
        } finally {
            setLoading(false);
            setCurrentlyProcessing('');
        }
    };

    const downloadSingleImage = (imageUrl, filename, type) => {
        try {
            if (!imageUrl || typeof imageUrl !== 'string') { // Check if imageUrl is a valid string
                alert('Image URL is missing or invalid, cannot download.');
                console.error("Download failed: imageUrl is invalid", imageUrl);
                return;
            }
            const link = document.createElement('a');
            link.href = imageUrl;
            link.download = `${type}_${(filename || 'download').replace(/[^a-zA-Z0-9._-]/g, '_')}`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (error) {
            console.error('Download single image/video failed:', error);
            alert('Download failed. Check console for details.');
        }
    };

    const downloadAllImages = (type) => { // 'original' or 'processed'
        const validResultsToDownload = results.filter(r => !r.error && r[type]);
        if (validResultsToDownload.length === 0) {
            alert(`No valid ${type} files to download.`);
            return;
        }
        validResultsToDownload.forEach((result, index) => {
            setTimeout(() => { // Stagger downloads
                downloadSingleImage(result[type], result.filename, type);
            }, index * 300);
        });
    };
    
    const clearAll = () => {
        setFiles([]);
        setResults([]);
        setBulkTrainingData(null);
        setMessage('');
        setError('');
        if (fileInputRef.current) fileInputRef.current.value = '';
        stopCamera(); // This will also reset camera-related states
        console.log('All states cleared.');
    };

    const extractTrainingDataAll = async () => {
        if (files.length === 0) {
            alert('Please select and process some files first (images or videos).');
            return;
        }

        // No longer filter for only images here, send all files
        // const imageFilesForTraining = files.filter(f => f.type.startsWith('image/'));
        const filesForTraining = files; // Send all files

        if (filesForTraining.length === 0) {
            // This case should ideally not be hit if files.length > 0 check passes
            setMessage('ℹ️ No files selected to extract training data from.');
            alert('No files selected to extract training data from.');
            return;
        }

        setExtractingTrainingData(true);
        setMessage(`🎯 Extracting training data for all ${filesForTraining.length} selected file(s)...`);
        setBulkTrainingData(null);  
        setError('');

        try {
            const formData = new FormData();
            filesForTraining.forEach(file => {
                formData.append('files', file);
            });
            formData.append('confidence', confidenceThreshold.toString());

            setMessage(`🎯 Processing ${filesForTraining.length} file(s) for training data extraction... (videos may take longer)`);

            const response = await fetch('https://drone-detection-686868741947.europe-west1.run.app/api/extract-training-data-bulk', {
                method: 'POST',
                body: formData,
                mode: 'cors' // Ensure CORS is enabled
            });

            const responseData = await response.json();

            if (!response.ok) {
                throw new Error(responseData.error || `Bulk training data extraction failed with status ${response.status}`);
            }
            
            setBulkTrainingData({
                dataset_zip: responseData.dataset_zip,
                total_samples: responseData.extracted_count,
                files_with_detections_list: responseData.files_with_detections || [],
                files_without_detections_list: responseData.files_without_detections || [],
                confidence_threshold: responseData.confidence_threshold,
                files_included_count: (responseData.files_with_detections || []).length,
                dataset_size_mb: responseData.dataset_size_mb
            });
            
            const filesWithDetectionsCount = (responseData.files_with_detections || []).length;
            const filesWithoutDetectionsCount = (responseData.files_without_detections || []).length;
            
            setMessage(`✅ Training data extraction complete: ${responseData.extracted_count} samples from ${filesWithDetectionsCount}/${filesForTraining.length} files. ${filesWithoutDetectionsCount} files had no detections.`);

        } catch (error) {
            console.error('❌ Error in bulk training data extraction:', error);
            setMessage(`❌ Training data extraction failed: ${error.message}`);
            setError(`Training data extraction error: ${error.message}`);
            setBulkTrainingData(null);
        } finally {
            setExtractingTrainingData(false);
        }
    };

    const downloadBulkTrainingData = () => {
        if (!bulkTrainingData?.dataset_zip) {
            alert('No combined training data available for download. Please extract it first.');
            return;
        }

        try {
            const link = document.createElement('a');
            link.href = bulkTrainingData.dataset_zip;
            // Sanitize filename components
            const includedCount = bulkTrainingData.files_included_count || 0;
            const samplesCount = bulkTrainingData.total_samples || 0;
            const dateStr = new Date().toISOString().slice(0, 10);
            link.download = `combined_training_data_${includedCount}files_${samplesCount}samples_${dateStr}.zip`;
            
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            console.log(`📥 Combined training dataset download initiated: ${samplesCount} samples from ${includedCount} files`);
        } catch (error) {
            console.error('❌ Error downloading combined training data:', error);
            alert('Failed to download combined training data. Check console for details.');
        }
    };

    // Camera Functions (startCamera, stopCamera, handleCameraChange, takePhoto, startRecording, stopRecording, formatRecordingTime are assumed to be here from previous steps)
    // Ensure startCamera, stopCamera, handleCameraChange are defined as per previous steps
    // Ensure takePhoto, startRecording, stopRecording, formatRecordingTime are defined as per previous steps
    
    const startCamera = async (explicitDeviceIdFromArgs = null) => {
        try {
            setCameraError('');
            setIsCameraViewfinderReady(false);

            let explicitDeviceId = null;
            if (typeof explicitDeviceIdFromArgs === 'string' && explicitDeviceIdFromArgs.trim() !== '') {
                explicitDeviceId = explicitDeviceIdFromArgs;
            } else if (explicitDeviceIdFromArgs !== null && explicitDeviceIdFromArgs !== undefined && explicitDeviceIdFromArgs !== '') {
                // Log if it's not null/undefined/empty but also not a string
                console.warn(`[startCamera V4] explicitDeviceIdFromArgs was not a valid string but was truthy:`, explicitDeviceIdFromArgs, `(type: ${typeof explicitDeviceIdFromArgs}). Treating as null.`);
            }
            
            let currentSelectedIdFromState = '';
            if (typeof selectedCameraId === 'string') {
                currentSelectedIdFromState = selectedCameraId;
            } else if (selectedCameraId && typeof selectedCameraId === 'object' && typeof selectedCameraId.deviceId === 'string') {
                console.warn(`[startCamera V4] selectedCameraId from state was an object, extracting .deviceId:`, selectedCameraId);
                currentSelectedIdFromState = selectedCameraId.deviceId;
            } else if (selectedCameraId) {
                 console.warn(`[startCamera V4] selectedCameraId from state was not a string or usable object:`, selectedCameraId, `(type: ${typeof selectedCameraId}). Treating as empty string.`);
            }

            console.log(`🔄 [startCamera V4] BEGIN. explicitDeviceId (cleaned): "${explicitDeviceId}", currentSelectedIdFromState (cleaned): "${currentSelectedIdFromState}"`);

            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Camera access not supported in this browser');
            }

            let constraintsToTry = [];
            const deviceIdToUseValue = explicitDeviceId || currentSelectedIdFromState;
            // Ensure deviceIdToUse is a non-empty string or null.
            const deviceIdToUse = (typeof deviceIdToUseValue === 'string' && deviceIdToUseValue.trim() !== '') ? deviceIdToUseValue : null;


            console.log(`[startCamera V4] Determined deviceIdToUse: "${deviceIdToUse}" (type: ${typeof deviceIdToUse})`);

            if (deviceIdToUse) { // This implies deviceIdToUse is a non-empty string
                console.log(`[startCamera V4] Trying specific device ID: "${deviceIdToUse}"`);
                constraintsToTry.push({ video: { deviceId: { exact: deviceIdToUse } }, audio: true });
                constraintsToTry.push({ video: { deviceId: { exact: deviceIdToUse } }, audio: false });
            } else {
                console.log('[startCamera V4] No specific valid camera ID. Trying generic fallbacks.');
                constraintsToTry.push({ video: { facingMode: 'environment' }, audio: true });
                constraintsToTry.push({ video: { facingMode: 'environment' }, audio: false });
                constraintsToTry.push({ video: { facingMode: 'user' }, audio: true });
                constraintsToTry.push({ video: { facingMode: 'user' }, audio: false });
                constraintsToTry.push({ video: true, audio: true }); // Simplest possible video constraint
                constraintsToTry.push({ video: true, audio: false });
            }
            
            let newStream = null;
            let lastError = null;
            
            for (let i = 0; i < constraintsToTry.length; i++) {
                try {
                    console.log(`[startCamera V4] Attempt ${i + 1}/${constraintsToTry.length} with constraints:`, constraintsToTry[i]);
                    newStream = await navigator.mediaDevices.getUserMedia(constraintsToTry[i]);
                    console.log('✅ [startCamera V4] Stream obtained:', newStream.id);
                    newStream.getVideoTracks().forEach(track => console.log('📹 Video track:', track.id, track.label, track.getSettings()));
                    newStream.getAudioTracks().forEach(track => console.log('🎤 Audio track:', track.id, track.label, track.getSettings()));
                    break; 
                } catch (error) {
                    console.warn(`❌ [startCamera V4] Constraint set ${i + 1} failed:`, error.name, error.message, constraintsToTry[i]);
                    lastError = error;
                }
            }
            
            if (!newStream) {
                console.error('❌ [startCamera V4] All constraint attempts failed. Last error:', lastError);
                throw lastError || new Error('Failed to access camera with all constraint attempts');
            }

            if (cameraStream && cameraStream.id !== newStream.id) {
                console.log('[startCamera V4] Stopping old stream:', cameraStream.id);
                cameraStream.getTracks().forEach(track => track.stop());
            }
            
            setCameraStream(newStream);
            setShowCamera(true);    
            console.log('[startCamera V4] Stream acquisition SUCCESS. useEffect will configure video element.');

            try {
                const devices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = devices.filter(device => device.kind === 'videoinput');
                setAvailableCameras(videoDevices);
                console.log('🕵️ [startCamera V4 DEBUG] availableCameras set. Count:', videoDevices.length, 'Example deviceId type:', videoDevices[0]?.deviceId ? typeof videoDevices[0].deviceId : 'N/A');

                const currentStreamVideoTrack = newStream?.getVideoTracks()?.[0];
                const actualDeviceIdFromStream = currentStreamVideoTrack?.getSettings?.().deviceId;

                let idToSelect = currentSelectedIdFromState; // Start with current (cleaned) state

                if (typeof actualDeviceIdFromStream === 'string' && actualDeviceIdFromStream.trim() !== '') {
                    const matchingEnumeratedDevice = videoDevices.find(d => d.deviceId === actualDeviceIdFromStream);
                    if (matchingEnumeratedDevice) {
                        idToSelect = actualDeviceIdFromStream; // Prefer actual stream ID if valid
                    } else {
                         console.warn(`[startCamera V4 Sync] actualDeviceIdFromStream (${actualDeviceIdFromStream}) not found in enumerated devices.`);
                    }
                } else {
                    console.log('[startCamera V4 Sync] No valid actualDeviceIdFromStream.');
                }
                
                // If idToSelect is still not a valid device from the list, or is empty, pick the first available one
                if (!idToSelect || !videoDevices.some(d => d.deviceId === idToSelect)) {
                    if (videoDevices.length > 0 && typeof videoDevices[0].deviceId === 'string') {
                         console.log(`[startCamera V4 Sync] idToSelect ("${idToSelect}") is invalid or empty. Defaulting to first enumerated: ${videoDevices[0].deviceId}`);
                        idToSelect = videoDevices[0].deviceId;
                    } else {
                        console.log(`[startCamera V4 Sync] No valid device to select for dropdown sync.`);
                        idToSelect = ''; // Fallback to empty if no valid devices
                    }
                }

                if (selectedCameraId !== idToSelect) { // selectedCameraId is the original state value here
                    console.log(`[startCamera V4 Sync] Updating selectedCameraId state from "${selectedCameraId}" to "${idToSelect}"`);
                    setSelectedCameraId(idToSelect);
                }

            } catch (enumError) {
                console.error('❌ [startCamera V4] Error during device enumeration or dropdown sync:', enumError);
            }
            
        } catch (error) {
            console.error('❌ [startCamera V4] Overall error in function:', error.name, error.message);
            if (cameraStream) { // Ensure any acquired stream is stopped on error
                cameraStream.getTracks().forEach(track => track.stop());
            }
            setCameraStream(null);
            setShowCamera(false);
            setIsCameraViewfinderReady(false);
            setAvailableCameras([]);
            let errorMessageText = 'Camera access failed: ';
            if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                errorMessageText += 'Permission denied. Please check browser settings.';
            } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
                errorMessageText += 'No camera found. Ensure it\'s connected and enabled.';
            } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
                errorMessageText += 'Camera is busy or unreadable. Try closing other apps using the camera.';
            } else if (error.name === 'OverconstrainedError' || error.name === 'ConstraintNotSatisfiedError') {
                errorMessageText += 'The selected camera settings are not supported by your device/browser. Try another camera or default settings.';
            } else {
                errorMessageText += error.message || 'Unknown camera error.';
            }
            setCameraError(errorMessageText);
        }
    };

    const stopCamera = () => {
        if (cameraStream) {
            console.log('🛑 Stopping camera stream and resetting camera state:', cameraStream.id);
            cameraStream.getTracks().forEach(track => {
                console.log(`   Stopping track: ${track.kind} - ${track.id} - ${track.label}`);
                track.stop();
            });
            setCameraStream(null);
        }
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
        }
        if (recordingTimer) {
            clearInterval(recordingTimer);
            setRecordingTimer(null);
        }
        setShowCamera(false);
        setIsRecording(false);
        setRecordingTime(0);
        setCameraError('');
        setIsCameraViewfinderReady(false);
        setAvailableCameras([]);
        setSelectedCameraId('');
        console.log('📷 Camera stopped and related states reset.');
    };

    const handleCameraChange = async (event) => {
        const valFromEvent = event.target.value;
        let newDeviceId = ''; // Default to empty string

        console.log(`[handleCameraChange V4] event.target.value: "${valFromEvent}" (type: ${typeof valFromEvent})`);

        if (typeof valFromEvent === 'string') {
            newDeviceId = valFromEvent;
        } else if (valFromEvent && typeof valFromEvent === 'object' && typeof valFromEvent.deviceId === 'string') {
            // This case is if event.target.value was unexpectedly an object (e.g., MediaDeviceInfo itself)
            // This should not happen with standard HTML select/option behavior where option value is a string.
            console.warn('[handleCameraChange V4] event.target.value was an object, extracting .deviceId. This is unexpected for a select onChange.');
            newDeviceId = valFromEvent.deviceId;
        } else {
            console.warn(`[handleCameraChange V4] event.target.value ("${valFromEvent}") is not a usable string. Treating as empty selection.`);
            // newDeviceId remains ''
        }

        // Ensure newDeviceId is a string, even if empty
        newDeviceId = String(newDeviceId);

        console.log(`[handleCameraChange V4] Determined newDeviceId to use: "${newDeviceId}" (type: ${typeof newDeviceId})`);
        
        // Update the state. This will also update the <select> value prop.
        setSelectedCameraId(newDeviceId); 

        if (cameraStream) {
            console.log('[handleCameraChange V4] Stopping current camera stream to switch. Stream ID:', cameraStream.id);
            cameraStream.getTracks().forEach(track => track.stop());
            setCameraStream(null); // Nullify the stream state
            setIsCameraViewfinderReady(false); // Reset viewfinder readiness
        }
        
        // Call startCamera with the explicitly selected (and cleaned) newDeviceId.
        // The slight delay helps ensure the old stream is fully released.
        console.log(`[handleCameraChange V4] Scheduling startCamera with explicitId: "${newDeviceId}" in 250ms`);
        setTimeout(async () => {
             console.log(`[handleCameraChange V4 TIMEOUT] Calling startCamera. explicitId: "${newDeviceId}". Current selectedCameraId state (for ref): "${selectedCameraId}"`);
             await startCamera(newDeviceId); 
        }, 250);
    };

    const takePhoto = () => {
        console.log('📸 Taking photo...');
        
        if (!cameraVideoRef.current || !canvasRef.current) {
            console.error('❌ Video or canvas element not ready');
            alert('Camera not ready. Please wait for the video to load.');
            return;
        }

        if (!cameraStream) {
            console.error('❌ No camera stream available');
            alert('Camera stream not available');
            return;
        }

        const video = cameraVideoRef.current;
        const canvas = canvasRef.current;

        let frameWidth = video.videoWidth;
        let frameHeight = video.videoHeight;

        if (frameWidth === 0 || frameHeight === 0) {
            console.warn('⚠️ Video element dimensions are 0. Using stream track settings as fallback.');
            const videoTrack = cameraStream.getVideoTracks()[0];
            if (videoTrack) {
                const trackSettings = videoTrack.getSettings();
                frameWidth = trackSettings.width || 0;
                frameHeight = trackSettings.height || 0;
            }
        }
        
        if (frameWidth === 0 || frameHeight === 0) {
            alert('Video not ready for photo. Try again.');
            return;
        }

        canvas.width = frameWidth;
        canvas.height = frameHeight;
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob((blob) => {
            if (blob) {
                const photoFile = new File([blob], `camera-photo-${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`, { type: 'image/jpeg' });
                setFiles(prevFiles => [...prevFiles, photoFile]);
                setMessage(`📸 Photo captured: ${photoFile.name}`);
                stopCamera(); // Close camera after photo
            } else {
                alert('Failed to capture photo blob.');
            }
        }, 'image/jpeg', 0.9);
    };

    const startRecording = () => {
        console.log('🎥 Starting recording...');
        if (!cameraStream || !cameraStream.active) { // Check stream active state
            alert('Camera stream not available or inactive. Please (re)start the camera.');
            return;
        }
        if (!window.MediaRecorder) {
            alert('MediaRecorder not supported.');
            return;
        }

        const codecs = [
            'video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 
            'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm',
            'video/mp4;codecs=h264,aac', 'video/mp4' // Added mp4 options
        ];
        let mimeType = '';
        for (const codec of codecs) {
            if (MediaRecorder.isTypeSupported(codec)) {
                mimeType = codec;
                break;
            }
        }
        console.log('✅ Using codec for recording:', mimeType || 'browser default');

        const options = { mimeType, videoBitsPerSecond: 1000000 }; // 1Mbps
        if (!mimeType) delete options.mimeType; // Let browser choose if no preference found

        try {
            const recorder = new MediaRecorder(cameraStream, options);
            let localRecordedChunks = []; // Use local variable for chunks

            recorder.ondataavailable = (event) => {
                if (event.data.size > 0) localRecordedChunks.push(event.data);
            };

            recorder.onstop = () => {
                const actualMimeType = recorder.mimeType || mimeType || 'video/webm'; // Ensure a valid MIME type
                const blob = new Blob(localRecordedChunks, { type: actualMimeType });
                const extension = actualMimeType.includes('mp4') ? '.mp4' : '.webm';
                const videoFile = new File([blob], `camera-video-${new Date().toISOString().replace(/[:.]/g, '-')}${extension}`, { type: actualMimeType });
                
                setFiles(prevFiles => [...prevFiles, videoFile]);
                setMessage(`🎥 Video recorded: ${videoFile.name} (${(videoFile.size / (1024*1024)).toFixed(2)}MB)`);
                
                localRecordedChunks = []; // Clear local chunks
                // setRecordedChunks([]); // Clear state chunks if they were used, now using local
                setIsRecording(false);
                // recordingTime is already reset by stopRecording/timer logic
                if (recordingTimer) { // Ensure timer is cleared if auto-stopped
                    clearInterval(recordingTimer);
                    setRecordingTimer(null);
                }
                 // Do not call stopCamera() here, let user decide or auto-stop after photo/video
            };

            recorder.onerror = (event) => {
                console.error('❌ MediaRecorder error:', event.error);
                alert(`Recording error: ${event.error.message || event.error.name || 'Unknown error'}`);
                setIsRecording(false);
                if (recordingTimer) {
                    clearInterval(recordingTimer);
                    setRecordingTimer(null);
                }
            };

            recorder.start();
            setMediaRecorder(recorder);
            setIsRecording(true);
            setRecordingTime(0); // Reset timer display
            
            const timer = setInterval(() => {
                setRecordingTime(prevTime => {
                    const newTime = prevTime + 1;
                    if (newTime >= MAX_RECORDING_TIME) {
                        if (recorder.state === 'recording') recorder.stop(); // Auto-stop
                        // Timer clearing is handled in onstop or if stopRecording is called
                        return MAX_RECORDING_TIME;
                    }
                    return newTime;
                });
            }, 1000);
            setRecordingTimer(timer);
        } catch (err) { // changed from error to err
             console.error('❌ Error setting up MediaRecorder:', err);
             alert(`Failed to start recording: ${err.message}`);
        }
    };

    const stopRecording = () => {
        if (mediaRecorder && mediaRecorder.state === "recording") { // Check state before stopping
            mediaRecorder.stop();
            // Timer is cleared in onstop or if it was running by MAX_RECORDING_TIME
        }
        // setIsRecording(false); // This is handled by onstop
        // setRecordingTime(0); // This is handled by onstop
    };

    const formatRecordingTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    // STYLES OBJECT (Should be complete as per your file)
    const styles = {
        container: { minHeight: '100vh', backgroundColor: '#f8fafc' },
        navbar: { backgroundColor: '#ffffff', padding: '1rem 0', borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, zIndex: 1000 },
        navContent: { maxWidth: '1200px', margin: '0 auto', padding: '0 2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
        logo: { fontSize: '1.5rem', fontWeight: '700', color: '#1e293b' },
        statusBadge: { padding: '0.5rem 1rem', borderRadius: '9999px', fontSize: '0.875rem', fontWeight: '500' /* Conditional bg/color in JSX */ },
        main: { maxWidth: '1200px', margin: '0 auto', padding: '3rem 2rem' },
        hero: { textAlign: 'center', marginBottom: '4rem' },
        title: { fontSize: '3rem', fontWeight: '800', color: '#1e293b', marginBottom: '1rem', lineHeight: '1.1' },
        subtitle: { fontSize: '1.25rem', color: '#64748b', fontWeight: '400', maxWidth: '600px', margin: '0 auto' },
        uploadCard: { backgroundColor: '#ffffff', borderRadius: '12px', padding: '2rem', boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)', border: '1px solid #e2e8f0', marginBottom: '3rem' },
        uploadArea: { border: dragActive ? '2px dashed #2563eb' : '2px dashed #cbd5e0', borderRadius: '8px', padding: '3rem 2rem', textAlign: 'center', cursor: 'pointer', transition: 'all 0.2s ease', backgroundColor: dragActive ? '#f0f9ff' : '#fafafa', marginBottom: '1.5rem', display: 'block', width: '100%', boxSizing: 'border-box' },
        uploadText: { fontSize: '1.125rem', fontWeight: '500', color: '#374151', marginBottom: '0.5rem', display: 'block' },
        uploadSubtext: { fontSize: '0.875rem', color: '#6b7280', display: 'block' },
        fileInput: { display: 'none' },
        fileName: { padding: '1rem', backgroundColor: '#dbeafe', color: '#1e40af', borderRadius: '6px', marginBottom: '1.5rem', fontSize: '0.875rem', fontWeight: '500', textAlign: 'center', wordBreak: 'break-word' },
        uploadButton: { backgroundColor: '#2563eb', color: '#ffffff', fontSize: '1rem', fontWeight: '600', padding: '0.75rem 2rem', borderRadius: '8px', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', width: '100%', transition: 'all 0.2s ease' },
        progressContainer: { margin: '1rem 0' },
        progressBar: { width: '100%', height: '8px', backgroundColor: '#e2e8f0', borderRadius: '4px', overflow: 'hidden', marginBottom: '0.5rem' },
        progressFill: { height: '100%', backgroundColor: '#2563eb', transition: 'width 0.3s ease' /* width is dynamic */ },
        progressText: { fontSize: '0.875rem', color: '#6b7280', textAlign: 'center' },
        bulkControls: { display: 'flex', gap: '1rem', marginTop: '1rem', flexWrap: 'wrap' },
        bulkButton: { backgroundColor: '#059669', color: '#ffffff', fontSize: '0.875rem', fontWeight: '600', padding: '0.5rem 1rem', borderRadius: '6px', border: 'none', cursor: 'pointer', transition: 'all 0.2s ease', flex: 1, minWidth: '120px' },
        clearButton: { backgroundColor: '#dc2626', color: '#ffffff', fontSize: '0.875rem', fontWeight: '600', padding: '0.5rem 1rem', borderRadius: '6px', border: 'none', cursor: 'pointer', transition: 'all 0.2s ease' },
        loadingSpinner: { width: '16px', height: '16px', border: '2px solid #ffffff', borderTop: '2px solid transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' },
        message: { padding: '1rem', borderRadius: '8px', margin: '1.5rem 0', fontWeight: '500' /* Conditional bg/color in JSX */ },
        resultsSection: { marginTop: '3rem' },
        resultsTitle: { fontSize: '2rem', fontWeight: '700', color: '#1e293b', marginBottom: '2rem', textAlign: 'center' },
        resultsGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '2rem' },
        imageCard: { backgroundColor: '#ffffff', borderRadius: '12px', padding: '1.5rem', boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)', border: '1px solid #e2e8f0', transition: 'all 0.2s ease' },
        imageTitle: { fontSize: '1rem', fontWeight: '600', color: '#1e293b', marginBottom: '1rem', wordBreak: 'break-word' },
        mediaContainer: { marginBottom: '1rem', display: 'flex', flexDirection: 'row', gap: '0.5rem' }, // Changed from column to row
        media: { width: '100%', maxWidth: '100%', /* Let flexbox handle sizing, removed maxWidth 150px */  height: 'auto', borderRadius: '8px', objectFit: 'contain' /* Changed from cover */ },
        downloadButton: { backgroundColor: '#059669', color: '#ffffff', fontSize: '0.875rem', fontWeight: '600', padding: '0.5rem 1rem', borderRadius: '6px', border: 'none', cursor: 'pointer', transition: 'all 0.2s ease', width: '100%' },
        errorMessage: { color: '#dc2626', fontSize: '0.875rem', fontStyle: 'italic', padding: '0.5rem', backgroundColor: '#fee2e2', borderRadius: '4px', marginTop: '0.5rem' },
        trainingSection: { backgroundColor: '#f8f4ff', borderRadius: '12px', padding: '2rem', margin: '2rem 0', border: '2px solid #e0e7ff' },
        trainingHeader: { fontSize: '1.5rem', fontWeight: '700', color: '#5b21b6', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' },
        trainingDescription: { fontSize: '0.875rem', color: '#6b7280', marginBottom: '1.5rem', lineHeight: '1.5' },
        sliderContainer: { marginBottom: '1.5rem' },
        sliderLabel: { display: 'block', fontSize: '0.875rem', fontWeight: '600', color: '#374151', marginBottom: '0.5rem' },
        slider: { width: '100%', height: '6px', borderRadius: '3px', background: '#e2e8f0', outline: 'none', appearance: 'none', WebkitAppearance: 'none' }, // Added WebkitAppearance
        sliderValue: { fontSize: '0.75rem', color: '#6b7280', marginTop: '0.25rem' },
        extractButton: { backgroundColor: '#7c3aed', color: '#ffffff', fontSize: '1rem', fontWeight: '600', padding: '0.75rem 2rem', borderRadius: '8px', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', width: '100%', transition: 'all 0.2s ease', marginBottom: '1rem' },
        toggleButton: { backgroundColor: '#6366f1', color: '#ffffff', fontSize: '0.875rem', fontWeight: '600', padding: '0.5rem 1rem', borderRadius: '6px', border: 'none', cursor: 'pointer', transition: 'all 0.2s ease', marginBottom: '2rem' },
        trainingResult: { backgroundColor: '#dcfce7', border: '1px solid #bbf7d0', borderRadius: '8px', padding: '1rem', marginTop: '1rem' },
        trainingResultText: { color: '#166534', fontSize: '0.875rem', fontWeight: '500', marginBottom: '0.5rem' },
        cameraSection: { backgroundColor: '#f0f9ff', borderRadius: '12px', padding: '2rem', margin: '2rem 0', border: '2px solid #0ea5e9' },
        cameraHeader: { fontSize: '1.5rem', fontWeight: '700', color: '#0c4a6e', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' },
        cameraControls: { display: 'flex', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap', justifyContent: 'center' },
        cameraButton: { backgroundColor: '#0ea5e9', color: '#ffffff', fontSize: '0.875rem', fontWeight: '600', padding: '0.75rem 1.5rem', borderRadius: '8px', border: 'none', cursor: 'pointer', transition: 'all 0.2s ease', flex: '1 1 auto', minWidth: '120px' }, // Allow flex grow/shrink
        recordButton: { backgroundColor: '#dc2626', color: '#ffffff', fontSize: '0.875rem', fontWeight: '600', padding: '0.75rem 1.5rem', borderRadius: '8px', border: 'none', cursor: 'pointer', transition: 'all 0.2s ease', flex: '1 1 auto', minWidth: '120px' },
        cameraContainer: { position: 'relative', backgroundColor: '#000000', borderRadius: '12px', overflow: 'hidden', marginBottom: '1rem', width: '100%', aspectRatio: '16/9' /* Modern way to set aspect ratio */ },
        cameraVideo: { display: 'block', width: '100%', height: '100%', objectFit: 'cover' },
        cameraOverlay: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(0, 0, 0, 0.5)', color: '#ffffff', fontSize: '1.125rem', fontWeight: '600' },
        recordingIndicator: { position: 'absolute', top: '1rem', right: '1rem', /* backgroundColor handled dynamically */ color: '#ffffff', padding: '0.5rem 1rem', borderRadius: '20px', fontSize: '0.875rem', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '0.5rem', animation: 'pulse 2s infinite' },
        hiddenCanvas: { display: 'none' }
    };

    return (
        <div style={styles.container}>
            <style>
                {`
                    @keyframes spin { to { transform: rotate(360deg); } }
                    .upload-area:hover { border-color: #2563eb; background-color: #f0f9ff; }
                    .upload-button:hover:not(:disabled) { background-color: #1d4ed8; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2); }
                    .upload-button:disabled { opacity: 0.6; cursor: not-allowed; }
                    .image-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); }
                    .download-button:hover { background-color: #047857; transform: translateY(-1px); }
                    .bulk-button:hover { background-color: #047857; transform: translateY(-1px); }
                    .clear-button:hover { background-color: #b91c1c; transform: translateY(-1px); }
                    .media:hover { transform: scale(1.02); }
                    .extract-button:hover:not(:disabled) { background-color: #6d28d9; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(124, 58, 237, 0.2); }
                    .toggle-button:hover { background-color: #4f46e5; transform: translateY(-1px); }
                    .camera-button:hover:not(:disabled) { background-color: #0284c7; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3); }
                    .record-button:hover:not(:disabled) { background-color: #b91c1c; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3); }
                    .camera-button:disabled, .record-button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; box-shadow: none; }
                    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
                    input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; height: 20px; width: 20px; border-radius: 50%; background: #7c3aed; cursor: pointer; border: 2px solid #ffffff; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); margin-top: -7px; /* Adjust for track height */ }
                    input[type="range"]::-moz-range-thumb { height: 20px; width: 20px; border-radius: 50%; background: #7c3aed; cursor: pointer; border: 2px solid #ffffff; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); }
                    input[type="range"]::-webkit-slider-runnable-track { background: #e2e8f0; height: 6px; border-radius: 3px; }
                    input[type="range"]::-moz-range-track { background: #e2e8f0; height: 6px; border-radius: 3px; border: none; }
                `}
            </style>

            <nav style={styles.navbar}>
                <div style={styles.navContent}>
                    <img src="/logo.png" alt="DroneDetect AI Logo" style={{ height: '40px' }} />
                    <div style={{...styles.statusBadge, backgroundColor: serverStatus === 'connected' ? '#dcfce7' : serverStatus === 'error' ? '#fee2e2' : '#fef9c3', color: serverStatus === 'connected' ? '#166534' : serverStatus === 'error' ? '#991b1b' : '#713f12' }}>
                        ● {serverStatus === 'connected' ? 'Connected' : serverStatus === 'error' ? 'Error' : 'Checking...'}
                    </div>
                </div>
            </nav>

            <main style={styles.main}>
                <div style={styles.hero}>
                    <h1 style={styles.title}>AI-Powered Drone Detection</h1>
                    <p style={styles.subtitle}>Upload unlimited images and one video. Our model detects drones with precision. Try direct camera capture for live analysis!</p>
                </div>

                <div style={styles.uploadCard}>
                    <div>
                        <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="image/*,video/*" multiple style={styles.fileInput} id="file-upload" />
                        <label htmlFor="file-upload" style={styles.uploadArea} className="upload-area" onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}>
                            <span style={styles.uploadText}>{dragActive ? '📁 Drop files here!' : '📁 Choose Images/Videos or Drag & Drop'}</span>
                            <span style={styles.uploadSubtext}>Unlimited Images (≤25MB ea) + 1 Video (≤75MB). Max 200MB total.</span>
                        </label>
                        
                        {files.length > 0 && (
                            <div style={styles.fileName}>
                                Selected {files.length} file(s): {files.map(f => `${f.name} (${(f.size / (1024*1024)).toFixed(1)}MB)`).join(', ')}
                            </div>
                        )}
                        
                        {loading && (
                            <div style={styles.progressContainer}>
                                <div style={styles.progressBar}><div style={{...styles.progressFill, width: `${processingProgress}%` }}></div></div>
                                <div style={styles.progressText}>Processing {currentlyProcessing || 'files'}... ({Math.round(processingProgress)}%)</div>
                            </div>
                        )}
                        
                        <button onClick={processAllFiles} disabled={loading || files.length === 0} style={styles.uploadButton} className="upload-button">
                            {loading && <span style={styles.loadingSpinner}></span>}
                            {loading ? `Processing...` : `📤 Analyze ${files.length || ''} File(s)`}
                        </button>

                        {results.length > 0 && !loading && ( // Only show if not loading and have results
                            <div style={styles.bulkControls}>
                                <button onClick={() => downloadAllImages('original')} style={styles.bulkButton} className="bulk-button">📥 All Originals</button>
                                <button onClick={() => downloadAllImages('processed')} style={styles.bulkButton} className="bulk-button">📥 All Processed</button>
                                <button onClick={clearAll} style={{...styles.clearButton, flexBasis: '100%', marginTop: '0.5rem'}} className="clear-button">🗑️ Clear All Files & Results</button>
                            </div>
                        )}
                         {files.length > 0 && !loading && results.length === 0 && ( // Show clear if files selected but no results yet
                             <div style={styles.bulkControls}>
                                <button onClick={clearAll} style={{...styles.clearButton, flexBasis: '100%', marginTop: '0.5rem'}} className="clear-button">🗑️ Clear Selection</button>
                             </div>
                         )}
                    </div>
                </div>

                <div style={styles.cameraSection}>
                    <h3 style={styles.cameraHeader}>📷 Camera Capture</h3>
                    <p style={{...styles.trainingDescription, color: '#0c4a6e', marginBottom: '1.5rem'}}>
                        Use your device camera for live analysis. Photos are processed immediately. Videos are limited to 720p and 2 minutes.
                    </p>
                    {!showCamera ? (
                        <div style={styles.cameraControls}><button onClick={startCamera} style={styles.cameraButton} className="camera-button" disabled={loading}>📷 Start Camera</button></div>
                    ) : (
                        <>
                            <div style={styles.cameraContainer}>
                                <video ref={cameraVideoRef} autoPlay playsInline muted style={styles.cameraVideo} />
                                {isRecording && (
                                    <div style={{...styles.recordingIndicator, backgroundColor: recordingTime > (MAX_RECORDING_TIME - 10) ? '#b91c1c' : recordingTime > (MAX_RECORDING_TIME - 20) ? '#f59e0b' : '#16a34a' }}>
                                        ● REC {formatRecordingTime(recordingTime)} / {formatRecordingTime(MAX_RECORDING_TIME)}
                                    </div>
                                )}
                                {(!cameraStream || !isCameraViewfinderReady) && !cameraError && ( // Show loading if stream not ready and no error
                                    <div style={styles.cameraOverlay}>Loading Camera...</div>
                                )}
                            </div>
                            
                            {showCamera && availableCameras.length > 1 && (
                                <div style={{ marginTop: '1rem', marginBottom: '1rem', textAlign: 'center' }}>
                                    <label htmlFor="camera-select" style={{ marginRight: '0.5rem', color: styles.cameraHeader.color, fontWeight: '600' }}>CAMERA:</label>
                                    <select id="camera-select" value={selectedCameraId} onChange={handleCameraChange} style={{ padding: '0.5rem 1rem', borderRadius: '6px', border: '1px solid #0ea5e9', backgroundColor: '#fff', color: '#0c4a6e', fontWeight: '500', cursor: 'pointer' }}>
                                        {availableCameras.map((device, index) => (
                                            <option key={device.deviceId} value={device.deviceId}>{device.label || `Camera ${index + 1}`}</option>
                                        ))}
                                    </select>
                                </div>
                            )}
                            
                            <div style={styles.cameraControls}>
                                <button onClick={takePhoto} style={styles.cameraButton} className="camera-button" disabled={!cameraStream || isRecording || !isCameraViewfinderReady || !!cameraError}>📸 Take Photo</button>
                                {!isRecording ? (
                                    <button onClick={startRecording} style={styles.recordButton} className="record-button" disabled={!cameraStream || !isCameraViewfinderReady || !!cameraError}>🎥 Start Recording</button>
                                ) : (
                                    <button onClick={stopRecording} style={{...styles.recordButton, backgroundColor: '#059669'}} className="record-button">🛑 Stop Recording</button>
                                )}
                                <button onClick={stopCamera} style={{...styles.cameraButton, backgroundColor: '#6b7280', flexBasis: '100%', marginTop: '0.5rem'}} className="camera-button">❌ Close Camera</button>
                            </div>
                        </>
                    )}
                    {cameraError && <div style={{ backgroundColor: '#fee2e2', color: '#dc2626', padding: '1rem', borderRadius: '8px', marginTop: '1rem', fontSize: '0.875rem' }}>{cameraError}</div>}
                    <canvas ref={canvasRef} style={styles.hiddenCanvas} />
                </div>

                <div style={styles.uploadCard}> {/* Training Data Section is inside an uploadCard for consistency */}
                    <button onClick={() => setShowTrainingExtraction(!showTrainingExtraction)} style={styles.toggleButton} className="toggle-button">
                        {showTrainingExtraction ? '🔼 Hide Training Data Extraction' : '🔽 Show Training Data Extraction'}
                    </button>
                    {showTrainingExtraction && (
                        <div style={styles.trainingSection}>
                            <h3 style={styles.trainingHeader}>🎯 Extract Training Dataset</h3>
                            <p style={styles.trainingDescription}>Generate YOLO format training data from uploaded images and videos. Creates a combined dataset from multiple files where drones are detected.</p>
                            <div style={styles.sliderContainer}>
                                <label style={styles.sliderLabel}>Confidence Threshold: {confidenceThreshold.toFixed(1)}</label>
                                <input type="range" min="0.1" max="0.9" step="0.1" value={confidenceThreshold} onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))} style={styles.slider} />
                                <div style={styles.sliderValue}>Higher values extract only high-confidence detections.</div>
                                </div>
                            <button onClick={extractTrainingDataAll} disabled={extractingTrainingData || files.length === 0} style={{...styles.extractButton, opacity: (extractingTrainingData || files.length === 0) ? 0.6 : 1}} className="extract-button">
                                {extractingTrainingData && <span style={styles.loadingSpinner}></span>}
                                {extractingTrainingData ? 'Extracting...' : `📦 Extract Dataset from ${files.length} File(s)`}
                            </button>
                            {bulkTrainingData && ( // Display for bulk training data
                                <div style={styles.trainingResult}>
                                    <div style={styles.trainingResultText}>Total Samples: {bulkTrainingData.total_samples} from {bulkTrainingData.files_included_count} files</div>
                                    {bulkTrainingData.files_without_detections_list && bulkTrainingData.files_without_detections_list.length > 0 && (
                                        <div style={{...styles.trainingResultText, color: '#d97706'}}>⚠️ {bulkTrainingData.files_without_detections_list.length} file(s) had no detections.</div>
                                    )}
                                    <div style={styles.trainingResultText}>Confidence Used: {bulkTrainingData.confidence_threshold.toFixed(1)}</div>
                                    <button onClick={downloadBulkTrainingData} style={{...styles.downloadButton, marginTop: '1rem', width: '100%', fontSize: '1rem', padding: '0.75rem'}}>
                                        📥 Download Combined Dataset ({bulkTrainingData.dataset_size_mb?.toFixed(2) || 'N/A'}MB)
                                    </button>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {error && <div style={{ ...styles.message, backgroundColor: '#fee2e2', color: '#dc2626' }}>{error}</div>}
                {message && !error && <div style={{ ...styles.message, backgroundColor: '#dcfce7', color: '#166534' }}>{message}</div>} {/* Only show general message if no error */}
                
                {results.length > 0 && (
                    <div style={styles.resultsSection}>
                        <h2 style={styles.resultsTitle}>Detection Results ({results.filter(r => !r.error).length} successful / {results.length} total)</h2>
                        <div style={styles.resultsGrid}>
                            {results.map((result) => (
                                <div key={result.id} style={styles.imageCard} className="image-card">
                                    <h3 style={styles.imageTitle}>{result.filename} <span style={{fontSize: '0.8rem', color: '#6b7280'}}>({result.fileSize}MB) [{result.fileType === 'video' ? '🎥' : '🖼️'}]</span></h3>
                                    {result.error ? (
                                        <p style={styles.errorMessage}>❌ Error: {result.error}</p>
                                    ) : (
                                        <>
                                            <div style={styles.mediaContainer}>
                                                {result.original && (
                                                    <div style={{flex: 1, textAlign: 'center'}}>
                                                        <p style={{fontSize: '0.75rem', color: '#6b7280', marginBottom: '0.5rem'}}>Original</p>
                                                        {result.fileType === 'video' ? <video src={result.original} controls style={styles.media} className="media"/> : <img src={result.original} alt={`Original ${result.filename}`} style={styles.media} className="media"/>}
                                                    </div>
                                                )}
                                                {result.processed && (
                                                    <div style={{flex: 1, textAlign: 'center'}}>
                                                        <p style={{fontSize: '0.75rem', color: '#6b7280', marginBottom: '0.5rem'}}>Processed</p>
                                                        {result.fileType === 'video' ? <video src={result.processed} controls style={styles.media} className="media"/> : <img src={result.processed} alt={`Processed ${result.filename}`} style={styles.media} className="media"/>}
                                </div>
                                        )}
                                    </div>
                                            <div style={{display: 'flex', gap: '0.5rem', marginTop: '1rem'}}>
                                                <button onClick={() => downloadSingleImage(result.original, result.filename, 'original')} style={{...styles.downloadButton, flex: 1}} className="download-button">📥 Original</button>
                                                <button onClick={() => downloadSingleImage(result.processed, result.filename, 'processed')} style={{...styles.downloadButton, flex: 1}} className="download-button">📥 Processed</button>
                                </div>
                                        </>
                            )}
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
};

export default FileUpload;