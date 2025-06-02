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
    const [trainingDataResults, setTrainingDataResults] = useState([]);
    const [bulkTrainingData, setBulkTrainingData] = useState(null);

    // Camera functionality state
    const [showCamera, setShowCamera] = useState(false);
    const [cameraStream, setCameraStream] = useState(null);
    const [isRecording, setIsRecording] = useState(false);
    const [mediaRecorder, setMediaRecorder] = useState(null);
    const [recordedChunks, setRecordedChunks] = useState([]);
    const [cameraError, setCameraError] = useState('');
    const [recordingTime, setRecordingTime] = useState(0);
    const [recordingTimer, setRecordingTimer] = useState(null);
    const cameraVideoRef = useRef(null);
    const canvasRef = useRef(null);
    const [isCameraViewfinderReady, setIsCameraViewfinderReady] = useState(false);

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
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                
                console.log('📡 Server test response status:', response.status);
                
                if (response.ok) {
                    const data = await response.json();
                    console.log('✅ Server test successful:', data);
                    setServerStatus('connected');
                } else {
                    console.error('❌ Server test failed with status:', response.status);
                    setServerStatus('error');
                }
            } catch (error) {
                console.error('❌ Server test error:', error);
                setServerStatus('error');
            }
        };
        
        testServer();
        
        // Test every 30 seconds to keep connection alive
        const interval = setInterval(testServer, 30000);
        return () => clearInterval(interval);
    }, []);

    // Cleanup camera on component unmount
    useEffect(() => {
        // This effect now runs when cameraStream changes, or on component unmount.
        // It ensures that if a cameraStream was active, its tracks are stopped.
        const currentStreamToClean = cameraStream; // Capture the stream instance at this effect's setup time.

        return () => {
            if (currentStreamToClean) {
                console.log('🧹 useEffect[cameraStream] cleanup: Stopping tracks for stream instance:', currentStreamToClean.id);
                let tracksStopped = false;
                currentStreamToClean.getTracks().forEach(track => {
                    if (track.readyState === 'live') {
                        track.stop();
                        tracksStopped = true;
                    }
                });
                if (tracksStopped) {
                    console.log('Tracks stopped for', currentStreamToClean.id);
                } else {
                    console.log('No live tracks to stop for', currentStreamToClean.id, 'or already stopped.');
                }
            }
        };
    }, [cameraStream]); // Now ONLY depends on cameraStream

    // Separate effect specifically for cleaning up the recordingTimer on component unmount as a safety net.
    // stopCamera and stopRecording should handle clearing it during active use.
    useEffect(() => {
        const timerIdToClearOnUnmount = recordingTimer; // Capture the timer ID at effect setup
        return () => {
            if (timerIdToClearOnUnmount) {
                console.log('🧹 useEffect[] unmount (safety net): Clearing recordingTimer ID:', timerIdToClearOnUnmount);
                clearInterval(timerIdToClearOnUnmount);
            }
        };
    }, []); // Empty dependency array: runs only on mount and unmount (cleanup on unmount)

    // Effect to setup video element when cameraStream is ready and showCamera is true
    useEffect(() => {
        if (showCamera && cameraStream && cameraVideoRef.current) {
            const video = cameraVideoRef.current;
            console.log('📹 useEffect [ caméra ] activé: Tentative de configuration de l\'élément vidéo.'); // French: useEffect [camera] active: Attempting to configure video element
            
            video.muted = true; // Ensure muted for autoplay
            console.log('📹 useEffect [ caméra ]: Vidéo mise en sourdine.', video.muted);
            
            console.log('📹 useEffect [ caméra ]: Assignation de cameraStream à srcObject...', cameraStream);
            video.srcObject = cameraStream;
            console.log('📹 useEffect [ caméra ]: srcObject assigné. Valeur actuelle:', video.srcObject);
            
            setIsCameraViewfinderReady(false); // Reset before attempting to load

            let resolved = false;
            const timeoutDuration = 15000; // 15 seconds timeout for video events

            const cleanupEventListeners = () => {
                console.log('🧹 useEffect [ caméra ]: Nettoyage des écouteurs d\'événements vidéo.'); // French: Cleaning up video event listeners
                video.removeEventListener('loadedmetadata', onLoadedMetadata);
                video.removeEventListener('canplay', onCanPlay);
                video.removeEventListener('playing', onPlaying);
                video.removeEventListener('stalled', onStalled);
                video.removeEventListener('error', onError);
            };

            const succeed = (eventName) => {
                if (resolved) return;
                resolved = true;
                cleanupEventListeners();
                console.log(`✅ Événement vidéo dans useEffect [ caméra ]: ${eventName}. Viseur prêt.`); // French: Video event in useEffect [camera]: Viewfinder ready
                setIsCameraViewfinderReady(true);
            };

            const fail = (errorMsg, errorObj) => {
                if (resolved) return;
                resolved = true;
                cleanupEventListeners();
                console.error(`❌ ${errorMsg} dans useEffect [ caméra ]`, errorObj || ''); // French: in useEffect [camera]
                setCameraError(`Erreur flux caméra: ${errorMsg.split('.')[0]}. Réessayez de démarrer la caméra.`); // French: Camera stream error: Try restarting camera
                setIsCameraViewfinderReady(false);
            };

            const onLoadedMetadata = () => {
                console.log('📹 Événement [ caméra ]: loadedmetadata. Dimensions vidéo:', video.videoWidth, 'x', video.videoHeight); // French: Event [camera]: Video dimensions
                // For debugging, let's try setting ready here, but ideally we wait for canplay/playing
                // succeed('loadedmetadata (debug ready)'); 
            };
            const onCanPlay = () => {
                console.log('📹 Événement [ caméra ]: canplay');
                succeed('canplay');
            };
            const onPlaying = () => {
                console.log('📹 Événement [ caméra ]: playing');
                succeed('playing');
            };
            const onStalled = () => console.warn('📹 Événement [ caméra ]: stalled (Problème réseau?)'); // French: Network issue?
            const onError = (event) => {
                console.error('📹 Événement [ caméra ]: error object:', event);
                let errorDetail = 'unknown error';
                if (event && event.target && event.target.error) {
                    const vidError = event.target.error;
                    errorDetail = `code ${vidError.code}, ${vidError.message}`;
                }
                fail(`Erreur élément vidéo flux (${errorDetail})`, event); // French: Video element stream error
            };

            video.addEventListener('loadedmetadata', onLoadedMetadata);
            video.addEventListener('canplay', onCanPlay);
            video.addEventListener('playing', onPlaying);
            video.addEventListener('stalled', onStalled);
            video.addEventListener('error', onError);
            
            console.log('📹 useEffect [ caméra ]: Tentative de video.play()'); // French: Attempting video.play()
            video.play().then(() => {
                console.log('✅ video.play() promesse résolue dans useEffect [ caméra ].'); // French: promise resolved in useEffect [camera]
                setTimeout(() => {
                    if (!resolved && video.readyState >= 3) { // HAVE_FUTURE_DATA or more
                        console.log('⏰ Délai dépassé [ caméra ]: Promesse Play résolue, readyState OK. Forçage viseur prêt.'); // French: Timeout [camera]: Play promise resolved, readyState OK. Forcing viewfinder ready
                        succeed('play_promise_and_readyState_OK');
                    }
                }, 1000);
            }).catch(playError => {
                console.warn('⚠️ video.play() promesse rejetée dans useEffect [ caméra ] (politique autoplay?):', playError.name, playError.message); // French: promise rejected in useEffect [camera] (autoplay policy?)
            });

            const timer = setTimeout(() => {
                if (!resolved) {
                   fail(`Temporisation configuration vidéo [ caméra ] après ${timeoutDuration / 1000}s. Dernier readyState: ${video.readyState}`); // French: Video setup timeout [camera] after ... Last readyState
                }
            }, timeoutDuration);

            return () => {
                console.log('🧹 Nettoyage useEffect [ caméra ]: Libération srcObject et écouteurs.'); // French: useEffect cleanup [camera]: Releasing srcObject and listeners
                clearTimeout(timer);
                cleanupEventListeners();
                if (video && video.srcObject) { // Check if video ref is still valid
                    console.log('🧹 Nettoyage useEffect [ caméra ]: Nullification de video.srcObject.'); // French: Nullifying video.srcObject
                    video.srcObject = null;
                }
                setIsCameraViewfinderReady(false);
            };
        } else if (!cameraStream && showCamera) {
            console.log('📹 useEffect [ caméra ]: cameraStream est nul mais showCamera est vrai. Assurer nettoyage.'); // French: cameraStream is null but showCamera is true. Ensure cleanup
            setIsCameraViewfinderReady(false);
        } else {
            console.log('📹 useEffect [ caméra ]: Conditions non remplies pour la configuration vidéo (showCamera OU cameraStream OU cameraVideoRef.current est faux).', 
                { showCamera, hasStream: !!cameraStream, hasVidRef: !!cameraVideoRef.current });
        }
    }, [cameraStream, showCamera]);

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
        }
    };

    const handleFileSelection = (selectedFiles) => {
        // Filter for valid image and video files
        const validImageTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
        const validVideoTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/webm'];
        const validTypes = [...validImageTypes, ...validVideoTypes];
        
        const validFiles = selectedFiles.filter(file => validTypes.includes(file.type));
        
        if (validFiles.length === 0) {
            setError('Please select valid files (Images: JPG, PNG, WEBP | Videos: MP4, AVI, MOV, WEBM)');
            return;
        }

        if (selectedFiles.length !== validFiles.length) {
            setError(`${selectedFiles.length - validFiles.length} files were skipped (only images and videos are supported)`);
        }

        // Separate images and videos
        const imageFiles = validFiles.filter(f => validImageTypes.includes(f.type));
        const videoFiles = validFiles.filter(f => validVideoTypes.includes(f.type));

        // Check video limit - only one video allowed
        if (videoFiles.length > 1) {
            setError('Only one video can be processed at a time. Please select one video and unlimited images.');
            return;
        }

        // Check total file size
        const totalSize = validFiles.reduce((sum, file) => sum + file.size, 0);
        const totalSizeMB = totalSize / (1024 * 1024);
        const maxTotalSizeMB = 200; // 200MB total limit

        if (totalSizeMB > maxTotalSizeMB) {
            setError(`Total file size too large (${totalSizeMB.toFixed(1)}MB). Maximum total size is ${maxTotalSizeMB}MB.`);
            return;
        }

        // Check individual file sizes with different limits for images vs videos
        const oversizedFiles = validFiles.filter(file => {
            const fileSizeMB = file.size / (1024 * 1024);
            const isVideo = validVideoTypes.includes(file.type);
            const maxSize = isVideo ? 75 : 25; // 75MB for videos, 25MB for images
            return fileSizeMB > maxSize;
        });
        
        if (oversizedFiles.length > 0) {
            const videoFiles = oversizedFiles.filter(f => validVideoTypes.includes(f.type));
            const imageFiles = oversizedFiles.filter(f => validImageTypes.includes(f.type));
            
            let errorMsg = 'Some files are too large: ';
            if (videoFiles.length > 0) {
                errorMsg += `Videos must be ≤75MB. `;
            }
            if (imageFiles.length > 0) {
                errorMsg += `Images must be ≤25MB.`;
            }
            setError(errorMsg);
            return;
        }

        setFiles(validFiles);
        setError('');
        
        const imageCount = imageFiles.length;
        const videoCount = videoFiles.length;
        
        let selectionMessage = '';
        if (imageCount > 0 && videoCount > 0) {
            selectionMessage = `✅ Selected ${imageCount} images + 1 video (${totalSizeMB.toFixed(1)}MB total)`;
        } else if (imageCount > 0) {
            selectionMessage = `✅ Selected ${imageCount} images (${totalSizeMB.toFixed(1)}MB total)`;
        } else {
            selectionMessage = `✅ Selected 1 video (${totalSizeMB.toFixed(1)}MB total)`;
        }
        
        console.log(selectionMessage);
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

        const newResults = [];

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
                    
                    // Use chunked upload for large videos (>25MB)
                    if (isVideo && fileSizeMB > 25) {
                        console.log('🔄 Using chunked upload for large video...');
                        const { processResponse } = await uploadLargeFile(file);
                        response = processResponse;
                    } else {
                        // Standard upload for images and small videos
                        const formData = new FormData();
                        formData.append('file', file);
                        
                        response = await fetch('https://drone-detection-686868741947.europe-west1.run.app/api/upload', {
                            method: 'POST',
                            body: formData
                        });
                    }

                    if (!response.ok) {
                        let errorMessage;
                        try {
                            const errorData = await response.json();
                            errorMessage = errorData.error || `HTTP ${response.status}`;
                        } catch {
                            errorMessage = `HTTP ${response.status}`;
                        }
                        throw new Error(errorMessage);
                    }

                    const data = await response.json();
                    
                    // Handle chunked download if needed
                    let originalUrl, processedUrl;
                    
                    if (data.chunked_download) {
                        console.log(`📥 Downloading chunked result for ${file.name}...`);
                        originalUrl = await downloadFileInChunks(data.original_file, 'original');
                        processedUrl = await downloadFileInChunks(data.processed_file, 'processed');
                    } else {
                        originalUrl = data.original_file || data.original;
                        processedUrl = data.output_file || data.processed;
                    }

                    newResults.push({
                        id: Date.now() + i,
                        filename: file.name,
                        original: originalUrl,
                        processed: processedUrl,
                        fileSize: fileSizeMB.toFixed(2),
                        fileType: isVideo ? 'video' : 'image'
                    });

                    setResults([...newResults]);
                    console.log(`✅ Completed ${file.name}`);

                } catch (fileError) {
                    console.error(`❌ Error processing ${file.name}:`, fileError);
                    newResults.push({
                        id: Date.now() + i,
                        filename: file.name,
                        error: fileError.message,
                        fileSize: fileSizeMB.toFixed(2),
                        fileType: isVideo ? 'video' : 'image'
                    });
                    setResults([...newResults]);
                }
            }

            setProcessingProgress(100);
            const successCount = newResults.filter(r => !r.error).length;
            const failCount = newResults.filter(r => r.error).length;
            const imageCount = newResults.filter(r => r.fileType === 'image' && !r.error).length;
            const videoCount = newResults.filter(r => r.fileType === 'video' && !r.error).length;
            
            setMessage(`✅ Completed processing ${files.length} files. ${successCount} successful (${imageCount} images, ${videoCount} videos), ${failCount} failed.`);

        } catch (error) {
            console.error('❌ Batch processing failed:', error);
            setMessage(`❌ Batch processing failed: ${error.message}`);
        } finally {
            setLoading(false);
            setCurrentlyProcessing('');
        }
    };

    const downloadFileInChunks = async (fileInfo, fileType) => {
        const chunks = [];
        
        try {
            console.log(`📥 Starting chunked download: ${fileInfo.chunks} chunks (${(fileInfo.size / (1024*1024)).toFixed(2)}MB)`);
            
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            for (let i = 0; i < fileInfo.chunks; i++) {
                const chunkUrl = `https://drone-detection-686868741947.europe-west1.run.app/api/download-chunk/${fileInfo.file_id}/${i}`;
                
                let retryCount = 0;
                const maxRetries = 3;
                let chunkSuccess = false;
                
                while (!chunkSuccess && retryCount < maxRetries) {
                    try {
                        if (retryCount > 0) {
                            const backoffDelay = Math.min(1000 * Math.pow(2, retryCount - 1), 8000);
                            await new Promise(resolve => setTimeout(resolve, backoffDelay));
                        }
                        
                        const response = await fetch(chunkUrl, {
                            method: 'GET',
                            mode: 'cors',
                            headers: {
                                'Accept': 'application/json',
                                'Content-Type': 'application/json'
                            }
                        });
                        
                        if (!response.ok) {
                            let errorMessage;
                            try {
                                const errorData = await response.json();
                                errorMessage = errorData.error || `HTTP ${response.status}`;
                            } catch (parseError) {
                                const errorText = await response.text();
                                errorMessage = errorText || `HTTP ${response.status}`;
                            }
                            throw new Error(`Failed to download chunk ${i + 1}: ${errorMessage}`);
                        }
                        
                        const chunkData = await response.json();
                        
                        if (!chunkData.chunk_data) {
                            throw new Error(`Chunk ${i + 1} has no data`);
                        }
                        
                        const binaryString = atob(chunkData.chunk_data);
                        const bytes = new Uint8Array(binaryString.length);
                        for (let j = 0; j < binaryString.length; j++) {
                            bytes[j] = binaryString.charCodeAt(j);
                        }
                        
                        chunks.push(bytes);
                        chunkSuccess = true;
                        
                    } catch (fetchError) {
                        retryCount++;
                        if (retryCount >= maxRetries) {
                            throw new Error(`Failed to download chunk ${i + 1} after ${maxRetries} attempts: ${fetchError.message}`);
                        }
                    }
                }
                
                if (i < fileInfo.chunks - 1) {
                    await new Promise(resolve => setTimeout(resolve, 300));
                }
            }
            
            const totalSize = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
            const combined = new Uint8Array(totalSize);
            
            let offset = 0;
            for (const chunk of chunks) {
                combined.set(chunk, offset);
                offset += chunk.length;
            }
            
            const blob = new Blob([combined], { type: fileInfo.mime_type });
            const objectUrl = URL.createObjectURL(blob);
            
            // Cleanup server file
            try {
                const cleanupUrl = `https://drone-detection-686868741947.europe-west1.run.app/api/cleanup-download/${fileInfo.file_id}`;
                await fetch(cleanupUrl, {
                    method: 'POST',
                    mode: 'cors',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
            } catch (cleanupError) {
                console.warn('⚠️ Failed to cleanup server file:', cleanupError);
            }
            
            return objectUrl;
            
        } catch (error) {
            console.error(`❌ Chunked download failed for ${fileType}:`, error);
            throw error;
        }
    };

    const uploadLargeFile = async (file) => {
        const CHUNK_SIZE = 25 * 1024 * 1024; // 25MB chunks
        const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
        
        console.log(`📦 Splitting ${(file.size / (1024 * 1024)).toFixed(1)}MB file into ${totalChunks} chunks`);
        
        const uploadId = Date.now().toString();
        
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
            
            console.log(`🔄 Uploading chunk ${chunkIndex + 1}/${totalChunks}`);
            setMessage(`🔄 Uploading chunk ${chunkIndex + 1}/${totalChunks} for ${file.name}...`);

            const response = await fetch('https://drone-detection-686868741947.europe-west1.run.app/api/upload-chunk', {
                method: 'POST',
                body: chunkFormData
            });

            if (!response.ok) {
                throw new Error(`Chunk ${chunkIndex + 1} upload failed`);
            }
        }
        
        // Process the complete file
        console.log('📋 Processing complete file...');
        setMessage(`🤖 Processing complete file ${file.name}... This may take several minutes...`);
        
        const controller = new AbortController();
        const processResponse = await fetch('https://drone-detection-686868741947.europe-west1.run.app/api/process-uploaded', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ uploadId, fileName: file.name }),
            signal: controller.signal
        });
        
        return { processResponse, controller };
    };

    const downloadSingleImage = (imageUrl, filename, type) => {
        try {
            const link = document.createElement('a');
            link.href = imageUrl;
            link.download = `${type}_${filename}`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (error) {
            alert('Download failed');
        }
    };

    const downloadAllImages = (type) => {
        const validResults = results.filter(r => !r.error);
        if (validResults.length === 0) {
            alert('No processed images to download');
            return;
        }

        validResults.forEach((result, index) => {
            setTimeout(() => {
                const imageUrl = type === 'original' ? result.original : result.processed;
                downloadSingleImage(imageUrl, result.filename, type);
            }, index * 200); // Stagger downloads
        });
    };

    const extractTrainingDataAll = async () => {
        if (files.length === 0) {
            alert('Please process some images first');
            return;
        }

        setExtractingTrainingData(true);
        setMessage('🎯 Extracting training data for all images...');
        setTrainingDataResults([]);
        setBulkTrainingData(null);

        try {
            // Use the new bulk extraction endpoint
            const formData = new FormData();
            
            // Only include image files (no videos for training data extraction)
            const imageFiles = files.filter(f => f.type.startsWith('image/'));
            
            if (imageFiles.length === 0) {
                setMessage('❌ No image files found for training data extraction');
                return;
            }
            
            // Add all image files to the form data
            imageFiles.forEach(file => {
                formData.append('files', file);
            });
            
            // Add confidence threshold
            formData.append('confidence', confidenceThreshold.toString());
            
            setMessage(`🎯 Processing ${imageFiles.length} images for training data extraction...`);
            
            const response = await fetch('https://drone-detection-686868741947.europe-west1.run.app/api/extract-training-data-bulk', {
                method: 'POST',
                body: formData,
                mode: 'cors'
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Bulk training data extraction failed');
            }

            const data = await response.json();
            
            // Set the bulk training data for download
            setBulkTrainingData({
                dataset_zip: data.dataset_zip,
                total_samples: data.extracted_count,
                files_included: data.files_with_detections,
                files_without_detections: data.files_without_detections,
                confidence_threshold: data.confidence_threshold,
                files_with_detections: data.files_with_detections.length,
                dataset_size_mb: data.dataset_size_mb
            });
            
            // Create results for display
            const combinedResults = [{
                filename: `Combined Dataset (${data.files_with_detections.length} files)`,
                extracted_count: data.extracted_count,
                confidence_threshold: data.confidence_threshold,
                files_included: data.files_with_detections,
                files_excluded: data.files_without_detections
            }];
            setTrainingDataResults(combinedResults);
            
            const filesWithDetections = data.files_with_detections.length;
            const filesWithoutDetections = data.files_without_detections.length;
            
            setMessage(`✅ Training data extraction complete: ${data.extracted_count} samples from ${filesWithDetections}/${imageFiles.length} images (${filesWithoutDetections} images had no detections)`);

        } catch (error) {
            console.error('❌ Error in bulk training data extraction:', error);
            setMessage(`❌ Training data extraction failed: ${error.message}`);
            setBulkTrainingData(null);
        } finally {
            setExtractingTrainingData(false);
        }
    };

    const downloadBulkTrainingData = () => {
        if (!bulkTrainingData?.dataset_zip) {
            alert('No combined training data available for download');
            return;
        }

        try {
            const link = document.createElement('a');
            link.href = bulkTrainingData.dataset_zip;
            link.download = `combined_training_data_${bulkTrainingData.files_included.length}files_${bulkTrainingData.total_samples}samples_${new Date().toISOString().slice(0, 10)}.zip`;
            
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            console.log(`📥 Combined training dataset download initiated: ${bulkTrainingData.total_samples} samples from ${bulkTrainingData.files_included.length} files`);
        } catch (error) {
            console.error('❌ Error downloading combined training data:', error);
            alert('Failed to download combined training data');
        }
    };

    const clearAll = () => {
        setFiles([]);
        setResults([]);
        setTrainingDataResults([]);
        setBulkTrainingData(null);
        setMessage('');
        setError('');
        stopCamera(); // Stop camera when clearing all
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    // Camera functionality
    const startCamera = async () => {
        try {
            setCameraError('');
            setCameraStream(null); // Reset stream before starting
            setIsCameraViewfinderReady(false);
            console.log('🔄 Starting camera acquisition...');
            
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Camera access not supported in this browser');
            }
            
            // Permission check can stay here, or be part of a pre-flight check
            try {
                const permissionStatus = await navigator.permissions.query({ name: 'camera' });
                console.log('📷 Camera permission status:', permissionStatus.state);
                if (permissionStatus.state === 'denied') {
                    throw new Error('Camera permission denied. Please enable camera access in browser settings.');
                }
            } catch (permError) {
                console.warn('⚠️ Permission API not fully supported, or error querying. Will proceed with getUserMedia.');
            }
            
            const constraints = [
                { video: true, audio: true }, // Try with audio first
                { video: true }, // Fallback to video only
                // More specific constraints if needed, but start simple
                {
                    video: { 
                        width: { ideal: 1280, max: 1280 },
                        height: { ideal: 720, max: 720 },
                        facingMode: 'environment',
                        frameRate: { ideal: 30, max: 30 }
                    }, 
                    audio: true 
                }
            ];
            
            let stream = null;
            let lastError = null;
            
            for (let i = 0; i < constraints.length; i++) {
                try {
                    console.log(`🔄 Trying camera constraint ${i + 1}:`, JSON.stringify(constraints[i]));
                    stream = await navigator.mediaDevices.getUserMedia(constraints[i]);
                    console.log('✅ Camera stream obtained with constraint', i + 1);
                    break;
                } catch (error) {
                    console.warn(`❌ Constraint ${i + 1} (${JSON.stringify(constraints[i])}) failed:`, error.name, error.message);
                    lastError = error;
                }
            }
            
            if (!stream) {
                throw lastError || new Error('Failed to access camera with all constraint attempts');
            }
            
            setCameraStream(stream); // Set the stream
            setShowCamera(true);     // THEN set showCamera to true to trigger rendering of video element
            console.log('✅ Camera stream acquired. Video element will now be set up via useEffect.');
            
        } catch (error) {
            console.error('❌ Camera acquisition error in startCamera:', error.name, error.message);
            setCameraStream(null);
            setShowCamera(false);
            setIsCameraViewfinderReady(false);
            
            let errorMessage = 'Camera access failed: ';
            if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                errorMessage += 'Permission denied. Please allow camera access and try again.';
            } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
                errorMessage += 'No camera found. Please connect a camera and try again.';
            } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
                errorMessage += 'Camera is busy or cannot be read. Please close other apps/tabs using the camera or try a different camera.';
            } else if (error.name === 'OverconstrainedError' || error.name === 'ConstraintNotSatisfiedError') {
                errorMessage += 'The selected camera settings are not supported by your device.';
            } else {
                errorMessage += error.message;
            }
            setCameraError(errorMessage);
        }
    };

    const stopCamera = () => {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
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
        setIsCameraViewfinderReady(false); // Reset when camera stops
        console.log('📷 Camera stopped');
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

        // Check if video has loaded and has dimensions
        let frameWidth = video.videoWidth;
        let frameHeight = video.videoHeight;

        if (frameWidth === 0 || frameHeight === 0) {
            console.warn('⚠️ Video element dimensions are 0. Attempting to use stream track settings as fallback.');
            const videoTrack = cameraStream.getVideoTracks()[0];
            if (videoTrack) {
                const trackSettings = videoTrack.getSettings();
                frameWidth = trackSettings.width || 0;
                frameHeight = trackSettings.height || 0;
                console.log(`📹 Fallback dimensions from track: ${frameWidth}x${frameHeight}`);
            }
        }
        
        if (frameWidth === 0 || frameHeight === 0) {
            console.error('❌ Video not loaded or has no dimensions even after fallback. Cannot take photo.');
            alert('Video not ready. Please wait for the camera to fully load and try again.');
            return;
        }

        console.log(`📸 Using dimensions for photo: ${frameWidth}x${frameHeight}`);
        console.log(`📹 Video ready state: ${video.readyState}`);
        console.log(`📹 Video paused: ${video.paused}`);

        try {
            const context = canvas.getContext('2d');

            // Set canvas dimensions to match video frame
            canvas.width = frameWidth;
            canvas.height = frameHeight;

            console.log(`🖼️ Canvas dimensions set to: ${canvas.width}x${canvas.height}`);

            // Draw current video frame to canvas
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Convert canvas to blob
            canvas.toBlob((blob) => {
                if (blob) {
                    // Create a file from the blob
                    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                    const photoFile = new File([blob], `camera-photo-${timestamp}.jpg`, { 
                        type: 'image/jpeg' 
                    });

                    // Add to files list
                    setFiles(prevFiles => [...prevFiles, photoFile]);
                    console.log(`📸 Photo captured: ${photoFile.name} (${(photoFile.size / (1024*1024)).toFixed(2)}MB)`);
                    
                    // Show success message
                    setMessage(`📸 Photo captured successfully: ${photoFile.name}`);
                    
                    // Close camera after taking photo
                    stopCamera();
                } else {
                    console.error('❌ Failed to create blob from canvas');
                    alert('Failed to capture photo. Please try again.');
                }
            }, 'image/jpeg', 0.9);
            
        } catch (error) {
            console.error('❌ Error capturing photo:', error);
            alert(`Photo capture failed: ${error.message}`);
        }
    };

    const startRecording = () => {
        console.log('🎥 Starting recording...');
        
        if (!cameraStream) {
            console.error('❌ Camera stream not available');
            alert('Camera not available');
            return;
        }

        try {
            // Check MediaRecorder support
            if (!window.MediaRecorder) {
                throw new Error('MediaRecorder not supported in this browser');
            }

            // Try different WebM codecs for better compression and compatibility
            const codecs = [
                'video/webm;codecs=vp9,opus',
                'video/webm;codecs=vp8,opus', 
                'video/webm;codecs=vp9',
                'video/webm;codecs=vp8',
                'video/webm',
                'video/mp4;codecs=h264,aac',
                'video/mp4'
            ];
            
            let mimeType = 'video/webm';
            for (const codec of codecs) {
                if (MediaRecorder.isTypeSupported(codec)) {
                    mimeType = codec;
                    console.log('✅ Using codec:', codec);
                    break;
                }
                console.log('❌ Codec not supported:', codec);
            }
            
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                console.warn('⚠️ No preferred codecs supported, using default');
                mimeType = ''; // Let browser choose
            }

            const options = {
                videoBitsPerSecond: 1000000 // 1Mbps - good quality but smaller files
            };
            
            if (mimeType) {
                options.mimeType = mimeType;
            }

            console.log('🎥 MediaRecorder options:', options);

            const recorder = new MediaRecorder(cameraStream, options);

            const chunks = [];
            
            recorder.ondataavailable = (event) => {
                console.log('📦 Data chunk received:', event.data.size, 'bytes');
                if (event.data.size > 0) {
                    chunks.push(event.data);
                }
            };

            recorder.onstop = () => {
                console.log('🛑 Recording stopped, processing...');
                const actualMimeType = recorder.mimeType || mimeType || 'video/webm';
                const blob = new Blob(chunks, { type: actualMimeType });
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                
                // Determine file extension
                let extension = '.webm';
                if (actualMimeType.includes('mp4')) {
                    extension = '.mp4';
                }
                
                const videoFile = new File([blob], `camera-video-${timestamp}${extension}`, { 
                    type: actualMimeType 
                });

                // Add to files list
                setFiles(prevFiles => [...prevFiles, videoFile]);
                const fileSizeMB = (videoFile.size / (1024*1024)).toFixed(2);
                console.log(`🎥 Video recorded: ${videoFile.name} (${fileSizeMB}MB) - Duration: ${recordingTime}s`);
                
                // Show success message
                setMessage(`🎥 Video recorded successfully: ${videoFile.name} (${fileSizeMB}MB, ${recordingTime}s)`);
                
                setRecordedChunks([]);
                setIsRecording(false);
                setRecordingTime(0);
                
                // Clear timer
                if (recordingTimer) {
                    clearInterval(recordingTimer);
                    setRecordingTimer(null);
                }
            };

            recorder.onerror = (event) => {
                console.error('❌ MediaRecorder error:', event.error);
                alert(`Recording error: ${event.error.message}`);
                setIsRecording(false);
                if (recordingTimer) {
                    clearInterval(recordingTimer);
                    setRecordingTimer(null);
                }
            };

            recorder.start();
            setMediaRecorder(recorder);
            setIsRecording(true);
            setRecordingTime(0);
            
            // Start timer
            const timer = setInterval(() => {
                setRecordingTime(prevTime => {
                    const newTime = prevTime + 1;
                    
                    // Auto-stop at max recording time
                    if (newTime >= MAX_RECORDING_TIME) {
                        recorder.stop();
                        clearInterval(timer);
                        setRecordingTimer(null);
                        console.log(`🛑 Recording auto-stopped at ${MAX_RECORDING_TIME} seconds`);
                        return MAX_RECORDING_TIME;
                    }
                    
                    return newTime;
                });
            }, 1000);
            
            setRecordingTimer(timer);
            console.log(`🎥 Recording started (max ${MAX_RECORDING_TIME}s, ${mimeType || 'default codec'})`);
            
        } catch (error) {
            console.error('❌ Recording error:', error);
            alert(`Recording failed: ${error.message}`);
        }
    };

    const stopRecording = () => {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            if (recordingTimer) {
                clearInterval(recordingTimer);
                setRecordingTimer(null);
            }
            console.log(`🛑 Recording stopped manually at ${recordingTime}s`);
        }
    };

    // Format recording time as MM:SS
    const formatRecordingTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    // Styles object
    const styles = {
        container: {
            minHeight: '100vh',
            backgroundColor: '#f8fafc'
        },
        navbar: {
            backgroundColor: '#ffffff',
            padding: '1rem 0',
            borderBottom: '1px solid #e2e8f0',
            position: 'sticky',
            top: 0,
            zIndex: 1000
        },
        navContent: {
            maxWidth: '1200px',
            margin: '0 auto',
            padding: '0 2rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
        },
        logo: {
            fontSize: '1.5rem',
            fontWeight: '700',
            color: '#1e293b'
        },
        statusBadge: {
            padding: '0.5rem 1rem',
            borderRadius: '9999px',
            fontSize: '0.875rem',
            fontWeight: '500',
            backgroundColor: '#dcfce7',
            color: '#166534'
        },
        main: {
            maxWidth: '1200px',
            margin: '0 auto',
            padding: '3rem 2rem'
        },
        hero: {
            textAlign: 'center',
            marginBottom: '4rem'
        },
        title: {
            fontSize: '3rem',
            fontWeight: '800',
            color: '#1e293b',
            marginBottom: '1rem',
            lineHeight: '1.1'
        },
        subtitle: {
            fontSize: '1.25rem',
            color: '#64748b',
            fontWeight: '400',
            maxWidth: '600px',
            margin: '0 auto'
        },
        uploadCard: {
            backgroundColor: '#ffffff',
            borderRadius: '12px',
            padding: '2rem',
            boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
            border: '1px solid #e2e8f0',
            marginBottom: '3rem'
        },
        uploadArea: {
            border: dragActive ? '2px dashed #2563eb' : '2px dashed #cbd5e0',
            borderRadius: '8px',
            padding: '3rem 2rem',
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            backgroundColor: dragActive ? '#f0f9ff' : '#fafafa',
            marginBottom: '1.5rem',
            display: 'block',
            width: '100%',
            boxSizing: 'border-box'
        },
        uploadText: {
            fontSize: '1.125rem',
            fontWeight: '500',
            color: '#374151',
            marginBottom: '0.5rem',
            display: 'block'
        },
        uploadSubtext: {
            fontSize: '0.875rem',
            color: '#6b7280',
            display: 'block'
        },
        fileInput: {
            display: 'none'
        },
        fileName: {
            padding: '1rem',
            backgroundColor: '#dbeafe',
            color: '#1e40af',
            borderRadius: '6px',
            marginBottom: '1.5rem',
            fontSize: '0.875rem',
            fontWeight: '500',
            textAlign: 'center',
            wordBreak: 'break-word'
        },
        uploadButton: {
            backgroundColor: '#2563eb',
            color: '#ffffff',
            fontSize: '1rem',
            fontWeight: '600',
            padding: '0.75rem 2rem',
            borderRadius: '8px',
            border: 'none',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.5rem',
            width: '100%',
            transition: 'all 0.2s ease'
        },
        progressContainer: {
            margin: '1rem 0'
        },
        progressBar: {
            width: '100%',
            height: '8px',
            backgroundColor: '#e2e8f0',
            borderRadius: '4px',
            overflow: 'hidden',
            marginBottom: '0.5rem'
        },
        progressFill: {
            height: '100%',
            backgroundColor: '#2563eb',
            transition: 'width 0.3s ease',
            width: `${processingProgress}%`
        },
        progressText: {
            fontSize: '0.875rem',
            color: '#6b7280',
            textAlign: 'center'
        },
        bulkControls: {
            display: 'flex',
            gap: '1rem',
            marginTop: '1rem',
            flexWrap: 'wrap'
        },
        bulkButton: {
            backgroundColor: '#059669',
            color: '#ffffff',
            fontSize: '0.875rem',
            fontWeight: '600',
            padding: '0.5rem 1rem',
            borderRadius: '6px',
            border: 'none',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            flex: 1,
            minWidth: '120px'
        },
        clearButton: {
            backgroundColor: '#dc2626',
            color: '#ffffff',
            fontSize: '0.875rem',
            fontWeight: '600',
            padding: '0.5rem 1rem',
            borderRadius: '6px',
            border: 'none',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
        },
        loadingSpinner: {
            width: '16px',
            height: '16px',
            border: '2px solid #ffffff',
            borderTop: '2px solid transparent',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite'
        },
        message: {
            padding: '1rem',
            borderRadius: '8px',
            margin: '1.5rem 0',
            fontWeight: '500'
        },
        resultsSection: {
            marginTop: '3rem'
        },
        resultsTitle: {
            fontSize: '2rem',
            fontWeight: '700',
            color: '#1e293b',
            marginBottom: '2rem',
            textAlign: 'center'
        },
        resultsGrid: {
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
            gap: '2rem'
        },
        imageCard: {
            backgroundColor: '#ffffff',
            borderRadius: '12px',
            padding: '1.5rem',
            boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
            border: '1px solid #e2e8f0',
            transition: 'all 0.2s ease'
        },
        imageTitle: {
            fontSize: '1rem',
            fontWeight: '600',
            color: '#1e293b',
            marginBottom: '1rem',
            wordBreak: 'break-word'
        },
        mediaContainer: {
            marginBottom: '1rem',
            display: 'flex',
            flexDirection: 'row',
            gap: '0.5rem'
        },
        media: {
            width: '100%',
            maxWidth: '150px',
            height: 'auto',
            borderRadius: '8px',
            transition: 'all 0.2s ease'
        },
        downloadButton: {
            backgroundColor: '#059669',
            color: '#ffffff',
            fontSize: '0.875rem',
            fontWeight: '600',
            padding: '0.5rem 1rem',
            borderRadius: '6px',
            border: 'none',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            width: '100%'
        },
        errorMessage: {
            color: '#dc2626',
            fontSize: '0.875rem',
            fontStyle: 'italic'
        },
        // Training data extraction styles
        trainingSection: {
            backgroundColor: '#f8f4ff',
            borderRadius: '12px',
            padding: '2rem',
            margin: '2rem 0',
            border: '2px solid #e0e7ff'
        },
        trainingHeader: {
            fontSize: '1.5rem',
            fontWeight: '700',
            color: '#5b21b6',
            marginBottom: '1rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
        },
        trainingDescription: {
            fontSize: '0.875rem',
            color: '#6b7280',
            marginBottom: '1.5rem',
            lineHeight: '1.5'
        },
        sliderContainer: {
            marginBottom: '1.5rem'
        },
        sliderLabel: {
            display: 'block',
            fontSize: '0.875rem',
            fontWeight: '600',
            color: '#374151',
            marginBottom: '0.5rem'
        },
        slider: {
            width: '100%',
            height: '6px',
            borderRadius: '3px',
            background: '#e2e8f0',
            outline: 'none',
            appearance: 'none'
        },
        sliderValue: {
            fontSize: '0.75rem',
            color: '#6b7280',
            marginTop: '0.25rem'
        },
        extractButton: {
            backgroundColor: '#7c3aed',
            color: '#ffffff',
            fontSize: '1rem',
            fontWeight: '600',
            padding: '0.75rem 2rem',
            borderRadius: '8px',
            border: 'none',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.5rem',
            width: '100%',
            transition: 'all 0.2s ease',
            marginBottom: '1rem'
        },
        toggleButton: {
            backgroundColor: '#6366f1',
            color: '#ffffff',
            fontSize: '0.875rem',
            fontWeight: '600',
            padding: '0.5rem 1rem',
            borderRadius: '6px',
            border: 'none',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            marginBottom: '2rem'
        },
        trainingResult: {
            backgroundColor: '#dcfce7',
            border: '1px solid #bbf7d0',
            borderRadius: '8px',
            padding: '1rem',
            marginTop: '1rem'
        },
        trainingResultText: {
            color: '#166534',
            fontSize: '0.875rem',
            fontWeight: '500',
            marginBottom: '0.5rem'
        },
        // Camera styles
        cameraSection: {
            backgroundColor: '#f0f9ff',
            borderRadius: '12px',
            padding: '2rem',
            margin: '2rem 0',
            border: '2px solid #0ea5e9'
        },
        cameraHeader: {
            fontSize: '1.5rem',
            fontWeight: '700',
            color: '#0c4a6e',
            marginBottom: '1rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
        },
        cameraControls: {
            display: 'flex',
            gap: '1rem',
            marginBottom: '1.5rem',
            flexWrap: 'wrap'
        },
        cameraButton: {
            backgroundColor: '#0ea5e9',
            color: '#ffffff',
            fontSize: '0.875rem',
            fontWeight: '600',
            padding: '0.75rem 1.5rem',
            borderRadius: '8px',
            border: 'none',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            flex: 1,
            minWidth: '120px'
        },
        recordButton: {
            backgroundColor: '#dc2626',
            color: '#ffffff',
            fontSize: '0.875rem',
            fontWeight: '600',
            padding: '0.75rem 1.5rem',
            borderRadius: '8px',
            border: 'none',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            flex: 1,
            minWidth: '120px'
        },
        cameraContainer: {
            position: 'relative',
            backgroundColor: '#000000',
            borderRadius: '12px',
            overflow: 'hidden',
            marginBottom: '1rem',
            aspectRatio: '16/9'
        },
        cameraVideo: {
            width: '100%',
            height: '100%',
            objectFit: 'cover'
        },
        cameraOverlay: {
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            color: '#ffffff',
            fontSize: '1.125rem',
            fontWeight: '600'
        },
        recordingIndicator: {
            position: 'absolute',
            top: '1rem',
            right: '1rem',
            backgroundColor: '#dc2626',
            color: '#ffffff',
            padding: '0.5rem 1rem',
            borderRadius: '20px',
            fontSize: '0.875rem',
            fontWeight: '600',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            animation: 'pulse 2s infinite'
        },
        hiddenCanvas: {
            display: 'none'
        }
    };

    return (
        <div style={styles.container}>
            <style>
                {`
                    @keyframes spin {
                        to { transform: rotate(360deg); }
                    }
                    
                    .upload-area:hover {
                        border-color: #2563eb;
                        background-color: #f0f9ff;
                    }
                    
                    .upload-button {
                        transition: all 0.2s ease;
                    }
                    
                    .upload-button:hover:not(:disabled) {
                        background-color: #1d4ed8;
                        transform: translateY(-1px);
                        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
                    }
                    
                    .upload-button:disabled {
                        opacity: 0.6;
                        cursor: not-allowed;
                    }
                    
                    .image-card:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                    }
                    
                    .download-button:hover {
                        background-color: #047857;
                        transform: translateY(-1px);
                    }
                    
                    .bulk-button:hover {
                        background-color: #047857;
                        transform: translateY(-1px);
                    }
                    
                    .clear-button:hover {
                        background-color: #b91c1c;
                        transform: translateY(-1px);
                    }
                    
                    .media:hover {
                        transform: scale(1.02);
                    }
                    
                    .extract-button:hover:not(:disabled) {
                        background-color: #6d28d9;
                        transform: translateY(-1px);
                        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.2);
                    }
                    
                    .toggle-button:hover {
                        background-color: #4f46e5;
                        transform: translateY(-1px);
                    }
                    
                    .camera-button:hover:not(:disabled) {
                        background-color: #0284c7;
                        transform: translateY(-1px);
                        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
                    }
                    
                    .record-button:hover:not(:disabled) {
                        background-color: #b91c1c;
                        transform: translateY(-1px);
                        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
                    }
                    
                    .camera-button:disabled, .record-button:disabled {
                        opacity: 0.6;
                        cursor: not-allowed;
                        transform: none;
                        box-shadow: none;
                    }
                    
                    @keyframes pulse {
                        0%, 100% { opacity: 1; }
                        50% { opacity: 0.5; }
                    }
                    
                    input[type="range"] {
                        -webkit-appearance: none;
                        appearance: none;
                        background: transparent;
                        cursor: pointer;
                    }
                    
                    input[type="range"]::-webkit-slider-track {
                        background: #e2e8f0;
                        height: 6px;
                        border-radius: 3px;
                    }
                    
                    input[type="range"]::-webkit-slider-thumb {
                        -webkit-appearance: none;
                        appearance: none;
                        height: 20px;
                        width: 20px;
                        border-radius: 50%;
                        background: #7c3aed;
                        cursor: pointer;
                        border: 2px solid #ffffff;
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                    }
                    
                    input[type="range"]::-moz-range-track {
                        background: #e2e8f0;
                        height: 6px;
                        border-radius: 3px;
                        border: none;
                    }
                    
                    input[type="range"]::-moz-range-thumb {
                        height: 20px;
                        width: 20px;
                        border-radius: 50%;
                        background: #7c3aed;
                        cursor: pointer;
                        border: 2px solid #ffffff;
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                    }
                `}
            </style>

            {/* Navigation */}
            <nav style={styles.navbar}>
                <div style={styles.navContent}>
                    <div style={styles.logo}>
                        DroneDetect AI
                    </div>
                    <div style={styles.statusBadge}>
                        {serverStatus === 'connected' && '● Connected'}
                        {serverStatus === 'error' && '● Disconnected'}
                        {serverStatus === 'checking' && '● Checking...'}
                    </div>
                </div>
            </nav>

            {/* Main Content */}
            <main style={styles.main}>
                {/* Hero Section */}
                <div style={styles.hero}>
                    <h1 style={styles.title}>
                        AI-Powered Drone Detection
                    </h1>
                    <p style={styles.subtitle}>
                        Upload unlimited images and one video and let our advanced machine learning model detect and highlight drones with precision and accuracy.
                    </p>
                </div>

                {/* Upload Section */}
                <div style={styles.uploadCard}>
                    <div style={{ marginBottom: '2rem' }}>
                        <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleFileChange}
                            accept="image/*,video/*"
                            multiple
                            style={styles.fileInput}
                            id="file-upload"
                        />
                        <label 
                            htmlFor="file-upload" 
                            style={styles.uploadArea}
                            className="upload-area"
                            onDragEnter={handleDrag}
                            onDragLeave={handleDrag}
                            onDragOver={handleDrag}
                            onDrop={handleDrop}
                        >
                            <span style={styles.uploadText}>
                                {dragActive ? '📁 Drop files here!' : '📁 Choose Images/Videos or Drag & Drop'}
                            </span>
                            <span style={styles.uploadSubtext}>
                                Select unlimited images + max 1 video (Images: JPG, PNG, WEBP ≤25MB | Videos: MP4, AVI, MOV, WEBM ≤75MB) - Max 200MB total
                            </span>
                        </label>
                        
                        {files.length > 0 && (
                            <div style={styles.fileName}>
                                Selected {files.length} files: {files.map(f => `${f.name} (${(f.size / (1024*1024)).toFixed(1)}MB)`).join(', ')}
                            </div>
                        )}

                        {loading && (
                            <div style={styles.progressContainer}>
                                <div style={styles.progressBar}>
                                    <div style={styles.progressFill}></div>
                                </div>
                                <div style={styles.progressText}>
                                    Processing {currentlyProcessing}... ({Math.round(processingProgress)}%)
                                </div>
                            </div>
                        )}
                        
                        <button 
                            onClick={processAllFiles}
                            disabled={loading || files.length === 0}
                            style={styles.uploadButton}
                            className="upload-button"
                        >
                            {loading && <span style={styles.loadingSpinner}></span>}
                            {loading ? `Processing ${files.length} files...` : `📤 Analyze ${files.length} Files`}
                        </button>

                        {/* Bulk Controls */}
                        {results.length > 0 && (
                            <div style={styles.bulkControls}>
                                <button 
                                    onClick={() => downloadAllImages('original')}
                                    style={styles.bulkButton}
                                    className="bulk-button"
                                >
                                    📥 Download All Originals
                                </button>
                                <button 
                                    onClick={() => downloadAllImages('processed')}
                                    style={styles.bulkButton}
                                    className="bulk-button"
                                >
                                    📥 Download All Processed
                                </button>
                                <button 
                                    onClick={clearAll}
                                    style={styles.clearButton}
                                    className="clear-button"
                                >
                                    🗑️ Clear All
                                </button>
                            </div>
                        )}
                    </div>
                </div>

                {/* Camera Section */}
                <div style={styles.cameraSection}>
                    <h3 style={styles.cameraHeader}>
                        📷 Camera Capture
                    </h3>
                    
                    <p style={{...styles.trainingDescription, color: '#0c4a6e', marginBottom: '1.5rem'}}>
                        Take photos or record videos directly from your device camera for instant drone detection analysis.
                        Videos limited to 720p resolution and 2 minutes maximum duration for optimal processing.
                    </p>
                    
                    {!showCamera ? (
                        <div style={styles.cameraControls}>
                            <button 
                                onClick={startCamera}
                                style={styles.cameraButton}
                                className="camera-button"
                                disabled={loading}
                            >
                                📷 Start Camera
                            </button>
                        </div>
                    ) : (
                        <>
                            <div style={styles.cameraContainer}>
                                <video
                                    ref={cameraVideoRef}
                                    autoPlay
                                    playsInline
                                    muted
                                    style={styles.cameraVideo} // Revert to using the defined style
                                />
                                {isRecording && (
                                    <div style={{
                                        ...styles.recordingIndicator,
                                        backgroundColor: recordingTime > 100 ? '#dc2626' : recordingTime > 90 ? '#f59e0b' : '#dc2626'
                                    }}>
                                        ● REC {formatRecordingTime(recordingTime)} / {formatRecordingTime(MAX_RECORDING_TIME)}
                                    </div>
                                )}
                                {!cameraStream && (
                                    <div style={styles.cameraOverlay}>
                                        Starting camera...
                                    </div>
                                )}
                            </div>
                            
                            <div style={styles.cameraControls}>
                                <button 
                                    onClick={takePhoto}
                                    style={styles.cameraButton}
                                    className="camera-button"
                                    disabled={!cameraStream || isRecording || !isCameraViewfinderReady}
                                >
                                    📸 Take Photo
                                </button>
                                
                                {!isRecording ? (
                                    <button 
                                        onClick={startRecording}
                                        style={styles.recordButton}
                                        className="record-button"
                                        disabled={!cameraStream || !isCameraViewfinderReady}
                                    >
                                        🎥 Start Recording
                                    </button>
                                ) : (
                                    <button 
                                        onClick={stopRecording}
                                        style={{...styles.recordButton, backgroundColor: '#059669'}}
                                        className="record-button"
                                    >
                                        🛑 Stop Recording
                                    </button>
                                )}
                                
                                <button 
                                    onClick={stopCamera}
                                    style={{...styles.cameraButton, backgroundColor: '#6b7280'}}
                                    className="camera-button"
                                >
                                    ❌ Close Camera
                                </button>
                            </div>
                        </>
                    )}
                    
                    {cameraError && (
                        <div style={{
                            backgroundColor: '#fee2e2',
                            color: '#dc2626',
                            padding: '1rem',
                            borderRadius: '8px',
                            marginTop: '1rem',
                            fontSize: '0.875rem'
                        }}>
                            {cameraError}
                        </div>
                    )}
                    
                    {/* Hidden canvas for photo capture */}
                    <canvas ref={canvasRef} style={styles.hiddenCanvas} />
                </div>

                {/* Training Data Extraction Section */}
                <div style={styles.uploadCard}>
                    <button 
                        onClick={() => setShowTrainingExtraction(!showTrainingExtraction)}
                        style={styles.toggleButton}
                        className="toggle-button"
                    >
                        {showTrainingExtraction ? '🔼 Hide Training Data Extraction' : '🔽 Show Training Data Extraction'}
                    </button>

                    {showTrainingExtraction && (
                        <div style={styles.trainingSection}>
                            <h3 style={styles.trainingHeader}>
                                🎯 Extract Training Dataset
                            </h3>
                            <p style={styles.trainingDescription}>
                                Generate YOLO format training datasets from your uploaded images and videos. This will extract frames with 
                                drone detections and create properly formatted annotation files for training your own models.
                                Works with multiple files to create a comprehensive dataset.
                            </p>

                            <div style={styles.sliderContainer}>
                                <label style={styles.sliderLabel}>
                                    Confidence Threshold: {confidenceThreshold.toFixed(1)}
                                </label>
                                <input
                                    type="range"
                                    min="0.1"
                                    max="0.9"
                                    step="0.1"
                                    value={confidenceThreshold}
                                    onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                                    style={styles.slider}
                                />
                                <div style={styles.sliderValue}>
                                    Higher values extract only high-confidence detections
                                </div>
                            </div>

                            <button 
                                onClick={extractTrainingDataAll}
                                disabled={extractingTrainingData || files.filter(f => f.type.startsWith('image/')).length === 0}
                                style={{
                                    ...styles.extractButton,
                                    opacity: (extractingTrainingData || files.filter(f => f.type.startsWith('image/')).length === 0) ? 0.6 : 1,
                                    cursor: (extractingTrainingData || files.filter(f => f.type.startsWith('image/')).length === 0) ? 'not-allowed' : 'pointer'
                                }}
                                className="extract-button"
                            >
                                {extractingTrainingData && <span style={styles.loadingSpinner}></span>}
                                {extractingTrainingData ? 'Extracting Training Data...' : `📦 Extract Combined Training Dataset from ${files.filter(f => f.type.startsWith('image/')).length} Images`}
                            </button>

                            {trainingDataResults.length > 0 && (
                                <div style={styles.trainingResult}>
                                    <div style={styles.trainingResultText}>
                                        ✅ Extraction Complete: {trainingDataResults.length} files with detections
                                    </div>
                                    <div style={styles.trainingResultText}>
                                        Files included: {bulkTrainingData?.files_included?.join(', ') || 'None'}
                                    </div>
                                    <div style={styles.trainingResultText}>
                                        Total Samples: {bulkTrainingData?.total_samples || 0}
                                    </div>
                                    <div style={styles.trainingResultText}>
                                        Confidence Used: {confidenceThreshold.toFixed(1)}
                                    </div>
                                    {bulkTrainingData?.files_without_detections > 0 && (
                                        <div style={{...styles.trainingResultText, color: '#d97706'}}>
                                            ⚠️ {bulkTrainingData.files_without_detections} files excluded (no detections)
                                        </div>
                                    )}
                                    
                                    {/* Single combined download button */}
                                    {bulkTrainingData && (
                                        <button 
                                            onClick={downloadBulkTrainingData}
                                            style={{...styles.downloadButton, marginTop: '1rem', width: '100%', fontSize: '1rem', padding: '0.75rem'}}
                                        >
                                            📥 Download Combined Training Dataset ({bulkTrainingData.total_samples} samples from {bulkTrainingData.files_included.length} files)
                                        </button>
                                    )}
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Error Display */}
                {error && (
                    <div style={{
                        ...styles.message,
                        backgroundColor: '#fee2e2',
                        color: '#dc2626'
                    }}>
                        {error}
                    </div>
                )}

                {/* Message */}
                {message && (
                    <div style={{
                        ...styles.message,
                        backgroundColor: message.includes('Error') ? '#fee2e2' : '#dcfce7',
                        color: message.includes('Error') ? '#dc2626' : '#166534'
                    }}>
                        {message}
                    </div>
                )}

                {/* Results */}
                {results.length > 0 && (
                    <div style={styles.resultsSection}>
                        <h2 style={styles.resultsTitle}>
                            Detection Results ({results.length} files)
                        </h2>
                        
                        <div style={styles.resultsGrid}>
                            {results.map((result, index) => (
                                <div key={result.id} style={styles.imageCard} className="image-card">
                                    <h3 style={styles.imageTitle}>
                                        {result.filename} ({result.fileSize}MB) 
                                        <span style={{fontSize: '0.75rem', color: '#6b7280', marginLeft: '0.5rem'}}>
                                            [{result.fileType === 'video' ? '🎥 Video' : '🖼️ Image'}]
                                        </span>
                                    </h3>
                                    <div style={styles.mediaContainer}>
                                        {result.error ? (
                                            <p style={styles.errorMessage}>❌ {result.error}</p>
                                        ) : (
                                            <>
                                                {result.original && (
                                                    <div style={{flex: 1}}>
                                                        <p style={{fontSize: '0.75rem', color: '#6b7280', marginBottom: '0.5rem'}}>Original</p>
                                                        {result.fileType === 'video' ? (
                                                            <video 
                                                                src={result.original} 
                                                                controls
                                                                style={styles.media}
                                                                className="media"
                                                            />
                                                        ) : (
                                                            <img 
                                                                src={result.original} 
                                                                alt={`Original ${result.filename}`} 
                                                                style={styles.media}
                                                                className="media"
                                                            />
                                                        )}
                                                    </div>
                                                )}
                                                {result.processed && (
                                                    <div style={{flex: 1}}>
                                                        <p style={{fontSize: '0.75rem', color: '#6b7280', marginBottom: '0.5rem'}}>Processed</p>
                                                        {result.fileType === 'video' ? (
                                                            <video 
                                                                src={result.processed} 
                                                                controls
                                                                style={styles.media}
                                                                className="media"
                                                            />
                                                        ) : (
                                                            <img 
                                                                src={result.processed} 
                                                                alt={`Processed ${result.filename}`} 
                                                                style={styles.media}
                                                                className="media"
                                                            />
                                                        )}
                                                    </div>
                                                )}
                                            </>
                                        )}
                                    </div>
                                    
                                    {!result.error && (
                                        <div style={{display: 'flex', gap: '0.5rem'}}>
                                            <button 
                                                onClick={() => downloadSingleImage(result.original, result.filename, 'original')}
                                                style={{...styles.downloadButton, flex: 1}}
                                                className="download-button"
                                            >
                                                📥 Original
                                            </button>
                                            <button 
                                                onClick={() => downloadSingleImage(result.processed, result.filename, 'processed')}
                                                style={{...styles.downloadButton, flex: 1}}
                                                className="download-button"
                                            >
                                                📥 Processed
                                            </button>
                                        </div>
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