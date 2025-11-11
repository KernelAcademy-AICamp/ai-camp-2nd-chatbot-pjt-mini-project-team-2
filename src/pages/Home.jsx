import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './Home.css';

function Home() {
  const navigate = useNavigate();

  // 파일 업로드 관련 상태
  const [file, setFile] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // 업로드된 파일 목록 로드
  useEffect(() => {
    loadUploadedFiles();
  }, []);

  // 파일 선택 핸들러
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      setError(null);
    } else {
      setError('PDF 파일만 업로드 가능합니다.');
      setFile(null);
    }
  };

  // 파일 업로드
  const handleUpload = async () => {
    if (!file) {
      setError('파일을 선택해주세요.');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:3001/api/pdf/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await response.json();
      alert('파일 업로드 성공!');
      setFile(null);
      loadUploadedFiles();
    } catch (err) {
      setError('업로드 실패: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // 업로드된 파일 목록 로드
  const loadUploadedFiles = async () => {
    try {
      console.log('Loading uploaded files...');
      const response = await fetch('http://localhost:3001/api/pdf/uploaded-files');
      console.log('Load files response status:', response.status);

      if (response.ok) {
        const data = await response.json();
        console.log('Loaded files data:', data);
        console.log('Number of files:', data.files ? data.files.length : 0);
        setUploadedFiles(data.files || []);
      }
    } catch (err) {
      console.error('Failed to load files:', err);
    }
  };

  // 개별 파일 삭제
  const deleteFile = async (fileName) => {
    console.log('Attempting to delete file:', fileName);

    if (!window.confirm('이 파일을 삭제하시겠습니까?')) return;

    setLoading(true);
    setError(null);

    try {
      const url = `http://localhost:3001/api/pdf/delete-file/${encodeURIComponent(fileName)}`;
      console.log('Delete URL:', url);

      const response = await fetch(url, {
        method: 'DELETE'
      });

      console.log('Delete response status:', response.status);

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Delete failed:', errorData);
        throw new Error(errorData.detail || 'Failed to delete file');
      }

      const data = await response.json();
      console.log('Delete response:', data);

      // 파일 목록 새로고침
      console.log('Reloading file list...');
      await loadUploadedFiles();
      console.log('File list reloaded');

      alert('파일이 삭제되었습니다.');
    } catch (err) {
      console.error('Delete error:', err);
      setError('파일 삭제 실패: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="home-container">
      <div className="home-content">
        <h1 className="site-title">Q-Pass</h1>
        <p className="site-subtitle">국가기술자격 시험 준비를 도와드립니다</p>

        {/* PDF 업로드 섹션 */}
        <div className="upload-section">
          <h2 className="section-title">📤 학습 자료 업로드</h2>
          <p className="section-description">
            PDF 파일을 업로드하면 AI 분석, 질문답변, 퀴즈 생성 기능을 사용할 수 있습니다
          </p>

          {/* 에러 메시지 */}
          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}

          <div className="file-upload-area">
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileChange}
              className="file-input-hidden"
              id="file-input"
            />

            {!file ? (
              <label htmlFor="file-input" className="file-select-label">
                <div className="upload-icon">📄</div>
                <div className="upload-text">
                  <p className="upload-title">PDF 파일 선택하기</p>
                  <p className="upload-subtitle">클릭하여 파일을 선택하세요</p>
                </div>
              </label>
            ) : (
              <div className="file-selected-container">
                <div className="file-selected-info">
                  <span className="file-icon">📄</span>
                  <div className="file-details">
                    <p className="file-name">{file.name}</p>
                    <p className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                </div>
                <div className="file-actions">
                  <button
                    onClick={handleUpload}
                    disabled={loading}
                    className="action-btn primary"
                  >
                    {loading ? '업로드 중...' : '업로드'}
                  </button>
                  <button
                    onClick={() => {
                      setFile(null);
                      setError(null);
                    }}
                    disabled={loading}
                    className="action-btn secondary"
                  >
                    취소
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* 업로드된 파일 목록 */}
          <div className="uploaded-files-section">
            <div className="section-header">
              <h3>업로드된 파일 목록 ({uploadedFiles.length})</h3>
            </div>
            <div className="file-list">
              {uploadedFiles.map((f, index) => (
                <div key={index} className="file-item">
                  <span className="file-icon">📄</span>
                  <div className="file-info">
                    <p className="file-name">{f.display_name}</p>
                    <p className="file-meta">{f.state} • {f.uri}</p>
                  </div>
                  <button
                    onClick={() => deleteFile(f.name)}
                    disabled={loading}
                    className="delete-file-btn"
                    title="파일 삭제"
                  >
                    🗑️
                  </button>
                </div>
              ))}
              {uploadedFiles.length === 0 && (
                <p className="no-files">업로드된 파일이 없습니다</p>
              )}
            </div>
          </div>
        </div>

        {/* 기능 카드 */}
        <div className="feature-cards">
          <div
            className="feature-card"
            onClick={() => navigate('/study-plan')}
          >
            <div className="card-icon">📚</div>
            <h2 className="card-title">학습 계획</h2>
            <p className="card-description">
              시험 일정에 맞춰 맞춤형 학습 계획을 생성합니다
            </p>
          </div>

          <div
            className="feature-card"
            onClick={() => navigate('/document-analyzer')}
          >
            <div className="card-icon">📄</div>
            <h2 className="card-title">학습 자료 분석</h2>
            <p className="card-description">
              업로드한 PDF로 AI 분석, 질문답변, 퀴즈 생성을 받으세요
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home;
