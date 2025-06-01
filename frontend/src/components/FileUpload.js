// src/FileUpload.js
import React, { useState, useEffect, useRef } from 'react';

const FileUpload = () => {
    const [file, setFile] = useState(null);
    const [originalMedia, setOriginalMedia] = useState('');
    const [processedMedia, setProcessedMedia] = useState('');
    const [mediaType, setMediaType] = useState(''); // 'image' or 'video'
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState('');
    const [serverStatus, setServerStatus] = useState('checking');
    const [currentSessionId, setCurrentSessionId] = useState(null);
    const fileInputRef = useRef(null);
    const [error, setError] = useState('');
    
    // Training data extraction state
    const [extractingTrainingData, setExtractingTrainingData] = useState(false);
    const [showTrainingExtraction, setShowTrainingExtraction] = useState(false);
    const [confidenceThreshold, setConfidenceThreshold] = useState(0.3);
    const [trainingDataResult, setTrainingDataResult] = useState(null);

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

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            // Updated file size limit to match backend
            const maxSizeMB = 75; // Changed from 25 to 75
            const fileSizeMB = selectedFile.size / (1024 * 1024);
            
            if (fileSizeMB > maxSizeMB) {
                setError(`File too large. Maximum size is ${maxSizeMB}MB.`);
                setFile(null);
                return;
            }
            
            setFile(selectedFile);
            setError('');
            
            // Create preview URL
            const url = URL.createObjectURL(selectedFile);
            setOriginalMedia(url);
        }
    };

    const handleUpload = async () => {
        if (!file) {
            alert('Please select a file first');
            return;
        }

        console.log('🚀 Starting file upload process...');

        // File validation
        const validImageTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
        const validVideoTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/webm'];
        const maxSize = 75 * 1024 * 1024; // 75MB
        const fileSizeMB = file.size / (1024 * 1024);

        if (file.size > maxSize) {
            alert('File too large! Maximum size is 75MB.');
            return;
        }

        if (!validImageTypes.includes(file.type) && !validVideoTypes.includes(file.type)) {
            alert('Invalid file type! Please select JPG, PNG, WEBP images or MP4, AVI, MOV, WEBM videos.');
            return;
        }

        setLoading(true);
        const currentMediaType = file.type.startsWith('video/') ? 'video' : 'image';
        setMediaType(currentMediaType);
        
        // Dynamic message based on file size
        if (currentMediaType === 'video') {
            if (fileSizeMB > 50) {
                setMessage(`🔄 Processing large video (${fileSizeMB.toFixed(1)}MB)... This may take 20-30 minutes. Please be patient and keep this tab open...`);
            } else if (fileSizeMB > 30) {
                setMessage(`🔄 Processing video (${fileSizeMB.toFixed(1)}MB)... This may take 15-20 minutes. Please wait...`);
            } else if (fileSizeMB > 15) {
                setMessage(`🔄 Processing video (${fileSizeMB.toFixed(1)}MB)... This may take 8-12 minutes. Please wait...`);
            } else {
                setMessage('🔄 Processing video... This may take 3-8 minutes depending on video length. Please wait...');
            }
        } else {
            setMessage(`🔄 Processing ${currentMediaType}...`);
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            let response;
            
            // Use chunked upload for files > 25MB
            if (fileSizeMB > 25) {
                console.log('🔄 Using chunked upload for large file...');
                const { processResponse, controller } = await uploadLargeFile(file);
                response = processResponse;
            } else {
                console.log('🔄 Using standard upload...');
                response = await fetch('https://drone-detection-686868741947.europe-west1.run.app/api/upload', {
                    method: 'POST',
                    body: formData
                });
            }
            
            console.log('📡 Response received - Status:', response.status);
            console.log('📡 Response headers:', [...response.headers.entries()]);

            if (!response.ok) {
                let errorMessage;
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.error || `HTTP ${response.status}`;
                } catch {
                    const errorText = await response.text();
                    errorMessage = errorText || `HTTP ${response.status}`;
                }
                throw new Error(errorMessage);
            }

            const data = await response.json();
            console.log('✅ File processing completed successfully');
            console.log('📊 Response data:', data);
            
            // Handle chunked download response
            if (data.chunked_download) {
                console.log('🔄 Processing chunked download response...');
                setMessage('📥 Downloading processed results...');
                
                // Download original file in chunks
                const originalUrl = await downloadFileInChunks(data.original_file, 'original');
                
                // Download processed file in chunks  
                const processedUrl = await downloadFileInChunks(data.processed_file, 'processed');
                
                setOriginalMedia(originalUrl);
                setProcessedMedia(processedUrl);
                setMessage(`✅ ${data.message} (File size: ${data.file_size_mb}MB) - Downloaded via chunked transfer`);
                
            } else {
                // Handle regular response (for smaller files)
                setOriginalMedia(data.original_file || data.original);
                setProcessedMedia(data.output_file || data.processed);
                setMessage(`✅ ${data.message} (File size: ${data.file_size_mb || fileSizeMB.toFixed(1)}MB)`);
            }
            
        } catch (error) {
            console.error('❌ File upload failed:', error);
            console.error('❌ Error details:', {
                name: error.name,
                message: error.message,
                stack: error.stack
            });
            
            // Enhanced error handling for different scenarios
            if (error.name === 'AbortError') {
                setMessage(`❌ Request timed out. Large files (${fileSizeMB.toFixed(1)}MB) need more time. Please try with a smaller file or check your connection.`);
            } else if (error.message.includes('NetworkError') || error.message.includes('fetch')) {
                setMessage('❌ Network error. Please check your internet connection and try again.');
            } else if (error.message.includes('CORS')) {
                setMessage('❌ Server configuration error (CORS). Please contact support.');
            } else if (error.message.includes('too large for response') || error.message.includes('Cloud Run limit')) {
                setMessage(`❌ ${error.message} The video was processed but the result is too large to display. Try with a shorter video (under 20MB).`);
            } else if (error.message.includes('File too large for processing')) {
                setMessage(`❌ ${error.message} Please compress your video or try with a shorter clip.`);
            } else if (error.message.includes('413')) {
                setMessage('❌ File or response too large. Please try with a smaller file (under 20MB for videos).');
            } else if (error.message.includes('Failed to download chunk')) {
                setMessage(`❌ Download failed: ${error.message}. The video was processed but couldn't be downloaded. Please try again.`);
            } else {
                setMessage(`❌ Error: ${error.message}`);
            }
        } finally {
            setLoading(false);
        }
    };

    const downloadFileInChunks = async (fileInfo, fileType) => {
        const chunks = []; // Declare chunks outside try block to avoid scope issues
        
        try {
            console.log(`📥 Starting chunked download: ${fileInfo.chunks} chunks (${(fileInfo.size / (1024*1024)).toFixed(2)}MB)`);
            console.log(`📋 File info:`, fileInfo);
            setMessage(`📥 Downloading ${fileType} result in ${fileInfo.chunks} chunks...`);
            
            // Add a small delay to give server time to prepare files after processing
            console.log(`⏳ Waiting 2 seconds for server to prepare files...`);
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // Download all chunks with retry logic
            for (let i = 0; i < fileInfo.chunks; i++) {
                setMessage(`📥 Downloading ${fileType} chunk ${i + 1}/${fileInfo.chunks}...`);
                
                const chunkUrl = `https://drone-detection-686868741947.europe-west1.run.app/api/download-chunk/${fileInfo.file_id}/${i}`;
                console.log(`📡 Requesting chunk ${i + 1} from: ${chunkUrl}`);
                
                // Retry logic for each chunk
                let retryCount = 0;
                const maxRetries = 3;
                let chunkSuccess = false;
                
                while (!chunkSuccess && retryCount < maxRetries) {
                    try {
                        if (retryCount > 0) {
                            console.log(`🔄 Retry ${retryCount}/${maxRetries} for chunk ${i + 1}`);
                            setMessage(`📥 Retrying ${fileType} chunk ${i + 1}/${fileInfo.chunks} (attempt ${retryCount + 1})...`);
                            // Exponential backoff: wait longer for each retry
                            const backoffDelay = Math.min(1000 * Math.pow(2, retryCount - 1), 8000); // Max 8 seconds
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
                        
                        console.log(`📡 Chunk ${i + 1} response status:`, response.status);
                        
                        if (!response.ok) {
                            // Try to get error details
                            let errorMessage;
                            try {
                                const errorData = await response.json();
                                errorMessage = errorData.error || `HTTP ${response.status}`;
                                // Only log error details on first attempt to reduce spam
                                if (retryCount === 0) {
                                    console.error(`❌ Chunk ${i + 1} error data:`, errorData);
                                }
                            } catch (parseError) {
                                if (retryCount === 0) {
                                    console.error(`❌ Failed to parse chunk ${i + 1} error:`, parseError);
                                }
                                const errorText = await response.text();
                                errorMessage = errorText || `HTTP ${response.status}`;
                                if (retryCount === 0) {
                                    console.error(`❌ Chunk ${i + 1} error text:`, errorText);
                                }
                            }
                            throw new Error(`Failed to download chunk ${i + 1}: ${errorMessage}`);
                        }
                        
                        const chunkData = await response.json();
                        console.log(`📦 Chunk ${i + 1} data received:`, {
                            chunk_index: chunkData.chunk_index,
                            total_chunks: chunkData.total_chunks,
                            chunk_size: chunkData.chunk_size,
                            data_length: chunkData.chunk_data ? chunkData.chunk_data.length : 0
                        });
                        
                        if (!chunkData.chunk_data) {
                            throw new Error(`Chunk ${i + 1} has no data`);
                        }
                        
                        // Convert base64 back to binary
                        try {
                            const binaryString = atob(chunkData.chunk_data);
                            const bytes = new Uint8Array(binaryString.length);
                            for (let j = 0; j < binaryString.length; j++) {
                                bytes[j] = binaryString.charCodeAt(j);
                            }
                            
                            chunks.push(bytes);
                            console.log(`📦 Successfully processed chunk ${i + 1}: ${bytes.length} bytes`);
                            chunkSuccess = true; // Mark as successful
                            
                        } catch (decodeError) {
                            console.error(`❌ Failed to decode chunk ${i + 1}:`, decodeError);
                            throw new Error(`Failed to decode chunk ${i + 1}: ${decodeError.message}`);
                        }
                        
                    } catch (fetchError) {
                        retryCount++;
                        // Only log detailed error info on first attempt to reduce spam
                        if (retryCount === 1) {
                            console.error(`❌ Network error for chunk ${i + 1} (attempt ${retryCount}):`, fetchError);
                        } else {
                            console.log(`🔄 Retry ${retryCount} failed for chunk ${i + 1}: ${fetchError.message}`);
                        }
                        
                        if (retryCount >= maxRetries) {
                            // Check if it's a CORS error
                            if (fetchError.message.includes('CORS') || fetchError.message.includes('NetworkError')) {
                                throw new Error(`CORS/Network error on chunk ${i + 1} after ${maxRetries} attempts. Server may not be responding properly.`);
                            }
                            
                            throw new Error(`Failed to download chunk ${i + 1} after ${maxRetries} attempts: ${fetchError.message}`);
                        }
                        
                        // Only log retry messages occasionally to reduce spam
                        if (retryCount <= 2) {
                            const backoffDelay = Math.min(1000 * Math.pow(2, retryCount - 1), 8000);
                            console.log(`⏳ Will retry chunk ${i + 1} in ${backoffDelay / 1000} seconds...`);
                        }
                    }
                }
                
                // Add longer delay between chunks to avoid overwhelming the server
                if (i < fileInfo.chunks - 1) {
                    await new Promise(resolve => setTimeout(resolve, 300)); // Increased from 100ms to 300ms
                }
            }
            
            // Combine all chunks
            console.log('🔧 Assembling chunks...');
            const totalSize = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
            const combined = new Uint8Array(totalSize);
            
            let offset = 0;
            for (const chunk of chunks) {
                combined.set(chunk, offset);
                offset += chunk.length;
            }
            
            // Create blob and object URL
            const blob = new Blob([combined], { type: fileInfo.mime_type });
            const objectUrl = URL.createObjectURL(blob);
            
            console.log(`✅ ${fileType} file assembled: ${(totalSize / (1024*1024)).toFixed(2)}MB`);
            
            // Cleanup server file
            try {
                const cleanupUrl = `https://drone-detection-686868741947.europe-west1.run.app/api/cleanup-download/${fileInfo.file_id}`;
                console.log(`🧹 Cleaning up server file: ${cleanupUrl}`);
                
                const cleanupResponse = await fetch(cleanupUrl, {
                    method: 'POST',
                    mode: 'cors',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                if (cleanupResponse.ok) {
                    console.log(`🧹 Cleaned up server file: ${fileInfo.file_id}`);
                } else {
                    console.warn(`⚠️ Cleanup failed for ${fileInfo.file_id}:`, cleanupResponse.status);
                }
            } catch (cleanupError) {
                console.warn('⚠️ Failed to cleanup server file:', cleanupError);
            }
            
            return objectUrl;
            
        } catch (error) {
            console.error(`❌ Chunked download failed for ${fileType}:`, error);
            
            // Add debug info
            console.log('🔍 Debug info for failed download:');
            console.log('   - File info:', fileInfo);
            console.log('   - Chunks downloaded:', chunks?.length || 0);
            console.log('   - Error details:', {
                name: error.name,
                message: error.message,
                stack: error.stack
            });
            
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
            setMessage(`🔄 Uploading chunk ${chunkIndex + 1}/${totalChunks}...`);
            
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
        setMessage('🤖 Processing complete video... This may take several minutes...');
        
        const controller = new AbortController();
        const processResponse = await fetch('https://drone-detection-686868741947.europe-west1.run.app/api/process-uploaded', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ uploadId, fileName: file.name }),
            signal: controller.signal
        });
        
        return { processResponse, controller };
    };

    const handleTrainingDataExtraction = async () => {
        if (!file) {
            alert('Please select a file first');
            return;
        }

        setExtractingTrainingData(true);
        setMessage('🎯 Extracting training data...');
        setError('');
        setTrainingDataResult(null);

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('confidence', confidenceThreshold.toString());

            console.log(`🎯 Starting training data extraction with confidence ${confidenceThreshold}`);

            const response = await fetch('https://drone-detection-686868741947.europe-west1.run.app/api/extract-training-data', {
                method: 'POST',
                body: formData,
                mode: 'cors'
            });

            console.log('📡 Training extraction response status:', response.status);

            if (!response.ok) {
                const errorData = await response.json();
                console.error('❌ Training extraction error:', errorData);
                throw new Error(errorData.error || errorData.message || 'Training data extraction failed');
            }

            const result = await response.json();
            console.log('✅ Training extraction successful:', result);

            setTrainingDataResult(result);
            setMessage(`✅ Training data extracted: ${result.extracted_count} samples (${result.dataset_size_mb}MB)`);

        } catch (error) {
            console.error('❌ Error extracting training data:', error);
            setError(error.message || 'Failed to extract training data. Please try again.');
            setMessage('');
        } finally {
            setExtractingTrainingData(false);
        }
    };

    const downloadTrainingData = () => {
        if (!trainingDataResult?.dataset_zip) {
            alert('No training data available for download');
            return;
        }

        try {
            // Create download link
            const link = document.createElement('a');
            link.href = trainingDataResult.dataset_zip;
            link.download = `drone_training_dataset_${new Date().toISOString().slice(0, 10)}.zip`;
            
            // Trigger download
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            console.log('📥 Training dataset download initiated');
        } catch (error) {
            console.error('❌ Error downloading training data:', error);
            alert('Failed to download training data');
        }
    };

    const handleCleanup = async () => {
        if (!currentSessionId) return;
        
        try {
            const response = await fetch(`https://drone-detection-686868741947.europe-west1.run.app/api/cleanup/${currentSessionId}`, {
                method: 'POST'
            });
            
            if (response.ok) {
                console.log('✅ Manual cleanup completed');
                setCurrentSessionId(null);
                setOriginalMedia(null);
                setProcessedMedia(null);
                setMessage('Files cleaned up successfully');
            }
        } catch (error) {
            console.error('❌ Cleanup failed:', error);
        }
    };

    const downloadMedia = (base64Data, filename) => {
        try {
            const link = document.createElement('a');
            link.href = base64Data;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (error) {
            alert('Download failed');
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
            border: '2px dashed #cbd5e0',
            borderRadius: '8px',
            padding: '3rem 2rem',
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            backgroundColor: '#fafafa',
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
            textAlign: 'center'
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
            gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
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
            fontSize: '1.25rem',
            fontWeight: '600',
            color: '#1e293b',
            marginBottom: '1rem'
        },
        mediaContainer: {
            marginBottom: '1rem'
        },
        media: {
            width: '100%',
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
                        Upload an image or video and let our advanced machine learning model detect and highlight drones with precision and accuracy.
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
                            style={styles.fileInput}
                            id="file-upload"
                        />
                        <label 
                            htmlFor="file-upload" 
                            style={styles.uploadArea}
                            className="upload-area"
                        >
                            <span style={styles.uploadText}>
                                📁 Choose File
                            </span>
                            <span style={styles.uploadSubtext}>
                                Drag and drop or click to select (JPG, PNG, WEBP, MP4)
                            </span>
                        </label>
                        
                        {file && (
                            <div style={styles.fileName}>
                                Selected: {file.name} ({(file.size / (1024*1024)).toFixed(2)} MB)
                            </div>
                        )}
                        
                        <button 
                            onClick={handleUpload}
                            disabled={loading || (!file)}
                            style={styles.uploadButton}
                            className="upload-button"
                        >
                            {loading && <span style={styles.loadingSpinner}></span>}
                            {loading ? `Processing ${mediaType}...` : '📤 Analyze File'}
                        </button>
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
                                Generate a YOLO format training dataset from your uploaded file. This will extract frames with 
                                drone detections and create properly formatted annotation files for training your own models.
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
                                onClick={handleTrainingDataExtraction}
                                disabled={extractingTrainingData || !file}
                                style={{
                                    ...styles.extractButton,
                                    opacity: (extractingTrainingData || !file) ? 0.6 : 1,
                                    cursor: (extractingTrainingData || !file) ? 'not-allowed' : 'pointer'
                                }}
                                className="extract-button"
                            >
                                {extractingTrainingData && <span style={styles.loadingSpinner}></span>}
                                {extractingTrainingData ? 'Extracting Training Data...' : '📦 Extract Training Dataset'}
                            </button>

                            {trainingDataResult && (
                                <div style={styles.trainingResult}>
                                    <div style={styles.trainingResultText}>
                                        ✅ Extraction Complete: {trainingDataResult.extracted_count} samples
                                    </div>
                                    <div style={styles.trainingResultText}>
                                        Dataset Size: {trainingDataResult.dataset_size_mb}MB
                                    </div>
                                    <div style={styles.trainingResultText}>
                                        Confidence Used: {trainingDataResult.confidence_threshold}
                                    </div>
                                    <button 
                                        onClick={downloadTrainingData}
                                        style={styles.downloadButton}
                                    >
                                        📥 Download Training Dataset
                                    </button>
                                </div>
                            )}
                        </div>
                    )}
                </div>

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
                {(originalMedia || processedMedia) && (
                    <div style={styles.resultsSection}>
                        <h2 style={styles.resultsTitle}>
                            Detection Results
                        </h2>
                        
                        <div style={styles.resultsGrid}>
                            {originalMedia && (
                                <div style={styles.imageCard} className="image-card">
                                    <h3 style={styles.imageTitle}>Original {mediaType}</h3>
                                    <div style={styles.mediaContainer}>
                                        {mediaType === 'image' ? (
                                            <img 
                                                src={originalMedia} 
                                                alt="Original" 
                                                style={styles.media}
                                                className="media"
                                            />
                                        ) : (
                                            <video 
                                                src={originalMedia} 
                                                controls
                                                style={styles.media}
                                                className="media"
                                            />
                                        )}
                                    </div>
                                    <button 
                                        onClick={() => downloadMedia(originalMedia, `original_${mediaType}.${mediaType === 'image' ? 'jpg' : 'mp4'}`)}
                                        style={styles.downloadButton}
                                        className="download-button"
                                    >
                                        Download Original
                                    </button>
                                </div>
                            )}

                            {processedMedia && (
                                <div style={styles.imageCard} className="image-card">
                                    <h3 style={styles.imageTitle}>Detection Result</h3>
                                    <div style={styles.mediaContainer}>
                                        {mediaType === 'image' ? (
                                            <img 
                                                src={processedMedia} 
                                                alt="Processed with drone detection" 
                                                style={styles.media}
                                                className="media"
                                            />
                                        ) : (
                                            <video 
                                                src={processedMedia} 
                                                controls
                                                style={styles.media}
                                                className="media"
                                            />
                                        )}
                                    </div>
                                    <button 
                                        onClick={() => downloadMedia(processedMedia, `processed_${mediaType}.${mediaType === 'image' ? 'jpg' : 'mp4'}`)}
                                        style={styles.downloadButton}
                                        className="download-button"
                                    >
                                        Download Result
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
};

export default FileUpload;