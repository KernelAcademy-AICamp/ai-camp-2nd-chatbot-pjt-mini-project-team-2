import React, { useState, useEffect } from 'react';
import './DocumentAnalyzer.css';

function DocumentAnalyzer() {
  // 마크다운 텍스트를 HTML로 변환하는 함수
  const renderMarkdown = (text) => {
    if (!text) return '';

    let html = text;

    // Bold 처리 (**text** or __text__)
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');

    // Italic 처리 (*text* or _text_)
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    html = html.replace(/_(.*?)_/g, '<em>$1</em>');

    // 코드 블록 처리 (```code```)
    html = html.replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>');

    // 인라인 코드 처리 (`code`)
    html = html.replace(/`(.*?)`/g, '<code>$1</code>');

    // 줄바꿈 처리
    html = html.split('\n').map(line => {
      // 헤딩 처리
      if (line.startsWith('### ')) {
        return '<h3>' + line.substring(4) + '</h3>';
      } else if (line.startsWith('## ')) {
        return '<h2>' + line.substring(3) + '</h2>';
      } else if (line.startsWith('# ')) {
        return '<h1>' + line.substring(2) + '</h1>';
      }

      // Bullet point 처리
      if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
        return '<li>' + line.trim().substring(2) + '</li>';
      }

      // 숫자 리스트 처리
      const numberMatch = line.trim().match(/^(\d+)\.\s(.+)/);
      if (numberMatch) {
        return '<li>' + numberMatch[2] + '</li>';
      }

      // 일반 줄
      return line.trim() ? '<p>' + line + '</p>' : '<br />';
    }).join('');

    // 리스트 항목들을 ul 태그로 감싸기
    html = html.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');

    return html;
  };
  const [activeTab, setActiveTab] = useState('upload');
  const [file, setFile] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [selectedFileUri, setSelectedFileUri] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // PDF 분석 관련 - 제거됨

  // RAG 관련
  const [chatMessage, setChatMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [conversationId, setConversationId] = useState('');

  // 퀴즈 관련
  const [quiz, setQuiz] = useState(null);
  const [numQuestions, setNumQuestions] = useState(5);
  const [difficulty, setDifficulty] = useState('medium');

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
      const response = await fetch('http://localhost:3001/api/pdf/uploaded-files');
      if (response.ok) {
        const data = await response.json();
        setUploadedFiles(data.files || []);
      }
    } catch (err) {
      console.error('Failed to load files:', err);
    }
  };


  // 채팅 메시지 전송
  const sendChatMessage = async () => {
    if (!chatMessage.trim()) return;

    if (uploadedFiles.length === 0) {
      setError('먼저 파일을 업로드해주세요.');
      return;
    }

    // 대화 ID 생성 (첫 메시지인 경우)
    const convId = conversationId || `conv_${Date.now()}`;
    if (!conversationId) {
      setConversationId(convId);
    }

    setLoading(true);
    setError(null);

    try {
      const fileUris = uploadedFiles.map(f => f.name);

      const response = await fetch('http://localhost:3001/api/rag/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: chatMessage,
          file_uris: fileUris,
          conversation_id: convId,
          model_name: 'gemini-2.5-flash'
        })
      });

      if (!response.ok) throw new Error('Failed to send message');

      const data = await response.json();

      setChatHistory([
        ...chatHistory,
        { role: 'user', content: chatMessage },
        { role: 'assistant', content: data.answer }
      ]);
      setChatMessage('');
    } catch (err) {
      setError('메시지 전송 실패: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // 퀴즈 생성
  const generateQuiz = async () => {
    if (uploadedFiles.length === 0) {
      setError('먼저 파일을 업로드해주세요.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:3001/api/rag/generate-quiz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_uri: uploadedFiles[0].name,
          num_questions: numQuestions,
          difficulty: difficulty,
          model_name: 'gemini-2.5-flash'
        })
      });

      if (!response.ok) throw new Error('Failed to generate quiz');

      const data = await response.json();
      setQuiz(data.quiz);
    } catch (err) {
      setError('퀴즈 생성 실패: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // 개별 파일 삭제
  const deleteFile = async (fileName) => {
    if (!window.confirm('이 파일을 삭제하시겠습니까?')) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`http://localhost:3001/api/pdf/delete-file/${fileName}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to delete file');
      }

      // 파일 목록 새로고침
      loadUploadedFiles();
    } catch (err) {
      setError('파일 삭제 실패: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container">
      <h1>학습 자료 분석</h1>
      <p>PDF 문서를 업로드하고 AI로 분석하세요</p>

      <div className="document-analyzer-container">
        {/* 탭 네비게이션 */}
        <div className="tab-navigation">
          <button
            className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            📤 업로드
          </button>
          <button
            className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            💭 Q&A 챗봇
          </button>
          <button
            className={`tab-btn ${activeTab === 'quiz' ? 'active' : ''}`}
            onClick={() => setActiveTab('quiz')}
          >
            📝 퀴즈
          </button>
        </div>

        {/* 에러 메시지 */}
        {error && (
          <div className="error-message">
            ⚠️ {error}
          </div>
        )}

        {/* 탭 콘텐츠 */}
        <div className="tab-content">
          {/* 업로드 탭 */}
          {activeTab === 'upload' && (
            <div className="tab-panel">
              <div className="upload-section">
                <h2>PDF 파일 업로드</h2>
                <div className="file-input-container">
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={handleFileChange}
                    className="file-input"
                  />
                  {file && (
                    <p className="selected-file">선택된 파일: {file.name}</p>
                  )}
                </div>

                <div className="action-buttons">
                  <button
                    onClick={handleUpload}
                    disabled={loading || !file}
                    className="action-btn primary"
                  >
                    {loading ? '업로드 중...' : '업로드'}
                  </button>
                </div>

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
            </div>
          )}

          {/* AI 챗봇 탭 */}
          {activeTab === 'chat' && (
            <div className="tab-panel">
              <h2>Q&A 챗봇</h2>
              <p style={{ fontSize: '0.9rem', color: '#666', marginTop: '-10px', marginBottom: '20px' }}>
                업로드한 문서를 기반으로 AI에게 자유롭게 질문해보세요
              </p>
              <div className="chat-section">
                <div className="chat-history">
                  {chatHistory.map((msg, index) => (
                    <div
                      key={index}
                      className={`chat-message ${msg.role}`}
                    >
                      <div className="message-label">
                        {msg.role === 'user' ? '👤 사용자' : '🤖 AI'}
                      </div>
                      <div
                        className="message-content"
                        dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }}
                      />
                    </div>
                  ))}
                  {chatHistory.length === 0 && (
                    <p className="empty-message">
                      AI 챗봇과 대화를 시작하세요!<br />
                      ex) 문서를 요약해주세요 혹은 ~에 대해 설명해주세요
                    </p>
                  )}
                </div>

                <div className="chat-input">
                  <textarea
                    value={chatMessage}
                    onChange={(e) => setChatMessage(e.target.value)}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendChatMessage();
                      }
                    }}
                    placeholder="메시지를 입력하세요..."
                    rows={3}
                    className="chat-textarea"
                  />
                  <button
                    onClick={sendChatMessage}
                    disabled={loading || !chatMessage.trim()}
                    className="send-btn"
                  >
                    {loading ? '전송 중...' : '전송'}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* 퀴즈 탭 */}
          {activeTab === 'quiz' && (
            <div className="tab-panel">
              <h2>문제 풀기</h2>
              <p style={{ fontSize: '0.9rem', color: '#666', marginTop: '-10px', marginBottom: '20px' }}>
                문서 내용을 기반으로 객관식 문제를 생성합니다. <br />
                문제 수와 난이도를 선택해주세요. 문제 수는 최대 10개까지 생성할 수 있습니다.
              </p>
              <div className="quiz-section">
                <div className="quiz-settings">
                  <div className="setting-item">
                    <label>문제 수:</label>
                    <input
                      type="number"
                      value={numQuestions}
                      onChange={(e) => setNumQuestions(parseInt(e.target.value))}
                      min="1"
                      max="10"
                      className="number-input"
                    />
                  </div>
                  <div className="setting-item">
                    <label>난이도:</label>
                    <select
                      value={difficulty}
                      onChange={(e) => setDifficulty(e.target.value)}
                      className="difficulty-select"
                    >
                      <option value="easy">쉬움</option>
                      <option value="medium">보통</option>
                      <option value="hard">어려움</option>
                    </select>
                  </div>
                  <button
                    onClick={generateQuiz}
                    disabled={loading || uploadedFiles.length === 0}
                    className="generate-btn"
                  >
                    {loading ? '퀴즈 생성 중...' : '퀴즈 생성'}
                  </button>
                </div>

                {quiz && (
                  <div className="quiz-result">
                    <h3>{quiz.quiz_title}</h3>
                    <p className="quiz-meta">
                      문제 수: {quiz.total_questions} | 난이도: {quiz.difficulty}
                    </p>
                    <div className="questions-list">
                      {quiz.questions && quiz.questions.map((q, index) => (
                        <div key={index} className="question-card">
                          <h4>문제 {q.question_number}</h4>
                          <div
                            className="question-text"
                            dangerouslySetInnerHTML={{ __html: renderMarkdown(q.question_text) }}
                          />
                          <div className="options">
                            {q.options && q.options.map((option, optIndex) => (
                              <div
                                key={optIndex}
                                className="option"
                                dangerouslySetInnerHTML={{ __html: renderMarkdown(option) }}
                              />
                            ))}
                          </div>
                          <div className="answer-section">
                            <p>
                              <strong>정답:</strong>{' '}
                              <span dangerouslySetInnerHTML={{ __html: renderMarkdown(q.correct_answer) }} />
                            </p>
                            <p>
                              <strong>해설:</strong>{' '}
                              <span dangerouslySetInnerHTML={{ __html: renderMarkdown(q.explanation) }} />
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default DocumentAnalyzer;
