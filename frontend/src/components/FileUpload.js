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
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
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