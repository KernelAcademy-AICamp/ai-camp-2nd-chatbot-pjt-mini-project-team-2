import React, { useState, useEffect, useRef } from 'react';
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
  const [activeTab, setActiveTab] = useState('chat');
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [selectedFileUri, setSelectedFileUri] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // RAG 관련
  const [chatMessage, setChatMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [conversationId, setConversationId] = useState('');

  // 퀴즈 설정
  const [difficulty, setDifficulty] = useState('medium');

  // PDF 퀴즈 챗봇 관련
  const [pdfQuizMessages, setPdfQuizMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: '안녕하세요! 학습 도우미 AI입니다 😊\n위에서 파일을 선택하고 설정을 완료한 후 "퀴즈 시작" 버튼을 누르면 문제를 만들어드릴게요!'
    }
  ]);
  const [pdfQuizInput, setPdfQuizInput] = useState('');
  const [pdfQuizLoading, setPdfQuizLoading] = useState(false);
  const [pdfQuizQuestions, setPdfQuizQuestions] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [isQuizMode, setIsQuizMode] = useState(false);
  const [isGeneratingMore, setIsGeneratingMore] = useState(false);

  const messagesContainerRef = useRef(null);

  // 업로드된 파일 목록 로드
  useEffect(() => {
    loadUploadedFiles();
  }, []);

  // PDF 퀴즈 챗봇 메시지 자동 스크롤
  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  }, [pdfQuizMessages]);

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

  // PDF 퀴즈 챗봇 헬퍼 함수
  const addPdfQuizBotMessage = (content) => {
    setPdfQuizMessages((prev) => [
      ...prev,
      { id: prev.length + 1, role: 'assistant', content }
    ]);
  };

  const addPdfQuizUserMessage = (content) => {
    setPdfQuizMessages((prev) => [
      ...prev,
      { id: prev.length + 1, role: 'user', content }
    ]);
  };

  const showNextQuestion = (qObj, questionNumber) => {
    const formatted = `📘 문제 ${questionNumber}\n${qObj.question}\n\n${qObj.options
      .map((opt, i) => `${i + 1}) ${opt}`)
      .join('\n')}`;
    addPdfQuizBotMessage(formatted);
  };

  // PDF 퀴즈 챗봇 메시지 전송
  const handlePdfQuizSendMessage = async () => {
    if (!pdfQuizInput.trim()) return;
    const message = pdfQuizInput.trim();
    addPdfQuizUserMessage(message);
    setPdfQuizInput('');

    // 퀴즈 모드 중이라면 정답 판별
    if (isQuizMode && pdfQuizQuestions.length > 0) {
      handleQuizAnswer(message);
      return;
    }

    // 퀴즈 모드가 아닐 때는 메시지 전송 불가
    addPdfQuizBotMessage('위에서 파일을 선택하고 "퀴즈 시작" 버튼을 눌러주세요!');
  };

  // 업로드된 파일로 퀴즈 생성 (Gemini API 직접 사용 - 5문제 배치)
  const handleGenerateQuizFromFile = async () => {
    if (!selectedFileUri) {
      setError('파일을 선택해주세요.');
      return;
    }

    setPdfQuizLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:3001/api/quiz/generate-from-uploaded', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_name: selectedFileUri,
          num_questions: 5, // 5문제 배치 생성
          difficulty: difficulty,
          question_type: 'multiple_choice'
        })
      });

      if (!response.ok) throw new Error('Failed to generate quiz');

      const data = await response.json();

      if (data.success && data.questions && data.questions.length > 0) {
        setPdfQuizQuestions(data.questions); // 5문제 모두 저장
        setCurrentQuestionIndex(0);
        setIsQuizMode(true);
        addPdfQuizBotMessage(`퀴즈를 시작합니다! 😄`);
        showNextQuestion(data.questions[0], 1);
      } else {
        addPdfQuizBotMessage('❌ 문제 생성에 실패했습니다. 다시 시도해주세요.');
      }
    } catch (err) {
      console.error('Quiz generation error:', err);
      setError('퀴즈 생성 실패: ' + err.message);
      addPdfQuizBotMessage('❌ 퀴즈 생성 중 오류가 발생했습니다. 다시 시도해주세요.');
    } finally {
      setPdfQuizLoading(false);
    }
  };

  // 추가 문제 배치 생성 (백그라운드에서 실행)
  const generateMoreQuestions = async () => {
    if (isGeneratingMore) return; // 이미 생성 중이면 중복 실행 방지

    setIsGeneratingMore(true);

    try {
      const response = await fetch('http://localhost:3001/api/quiz/generate-from-uploaded', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_name: selectedFileUri,
          num_questions: 5, // 5문제 배치 생성
          difficulty: difficulty,
          question_type: 'multiple_choice'
        })
      });

      if (!response.ok) throw new Error('Failed to generate quiz');

      const data = await response.json();

      if (data.success && data.questions && data.questions.length > 0) {
        // 기존 문제 배열에 새 문제들 추가
        setPdfQuizQuestions(prev => [...prev, ...data.questions]);
        console.log(`✅ 추가 ${data.questions.length}개 문제 생성 완료`);
      } else {
        console.error('❌ 추가 문제 생성 실패');
      }
    } catch (err) {
      console.error('More questions generation error:', err);
    } finally {
      setIsGeneratingMore(false);
    }
  };

  // 퀴즈 정답 처리 (무한 모드 - 배치 생성)
  const handleQuizAnswer = (answerText) => {
    const currentQ = pdfQuizQuestions[currentQuestionIndex];
    const correct = currentQ.answer.trim();
    const explanation = currentQ.explanation;
    const options = currentQ.options || [];

    // 사용자 답변 처리
    let userAnswer = answerText.trim();

    // 숫자 입력인 경우 (1, 2, 3, 4 또는 1번, 2번, 3번, 4번) 해당 선택지로 변환
    const numberMatch = userAnswer.match(/^([1-4])번?$/);
    if (numberMatch && options.length > 0) {
      const optionIndex = parseInt(numberMatch[1]) - 1;
      if (optionIndex >= 0 && optionIndex < options.length) {
        userAnswer = options[optionIndex];
      }
    }

    // 정답 체크 (대소문자 구분 없이, 공백 제거)
    const isCorrect =
      userAnswer.toLowerCase().includes(correct.toLowerCase()) ||
      correct.toLowerCase().includes(userAnswer.toLowerCase()) ||
      userAnswer.toLowerCase() === correct.toLowerCase();

    // 정답/오답 피드백
    if (isCorrect) {
      let feedback = '✅ 정답입니다! 잘하셨어요 👏';
      if (explanation) {
        feedback += `\n\n💡 해설: ${explanation}`;
      }
      addPdfQuizBotMessage(feedback);
    } else {
      let feedback = `❌ 오답이에요. 정답은 '${correct}'입니다.`;
      if (explanation) {
        feedback += `\n\n💡 해설: ${explanation}`;
      }
      addPdfQuizBotMessage(feedback);
    }

    // 다음 문제 인덱스
    const nextIndex = currentQuestionIndex + 1;

    // 다음 문제가 있으면 바로 표시
    if (nextIndex < pdfQuizQuestions.length) {
      setCurrentQuestionIndex(nextIndex);
      showNextQuestion(pdfQuizQuestions[nextIndex], nextIndex + 1);

      // 마지막에서 3번째 문제에 도달하면 백그라운드에서 추가 문제 생성 시작
      if (nextIndex === pdfQuizQuestions.length - 3 && !isGeneratingMore) {
        console.log('🔄 추가 문제 생성 시작...');
        generateMoreQuestions();
      }
    } else {
      // 문제가 더 이상 없는 경우 (추가 생성이 완료되지 않았을 때)
      addPdfQuizBotMessage('다음 문제를 준비하고 있습니다... ⏳');

      // 추가 문제 생성이 시작되지 않았다면 시작
      if (!isGeneratingMore) {
        generateMoreQuestions();
      }

      // 추가 문제가 생성될 때까지 대기 (최대 30초)
      const checkInterval = setInterval(() => {
        if (pdfQuizQuestions.length > nextIndex) {
          clearInterval(checkInterval);
          setCurrentQuestionIndex(nextIndex);
          showNextQuestion(pdfQuizQuestions[nextIndex], nextIndex + 1);
        }
      }, 500);

      // 타임아웃: 30초 후에는 interval 정리
      setTimeout(() => {
        clearInterval(checkInterval);
      }, 30000);
    }
  };

  return (
    <div className="page-container">
      <h1>학습 자료 분석</h1>
      <p>업로드된 PDF 문서를 기반으로 AI 분석, 질문답변, 퀴즈를 이용하세요</p>

      <div className="document-analyzer-container">
        {/* 탭 네비게이션 */}
        <div className="tab-navigation">
          <button
            className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            💭 Q&A
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
          {/* AI 챗봇 탭 */}
          {activeTab === 'chat' && (
            <div className="tab-panel">
              <h2>Q&A</h2>
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
              <h2>퀴즈</h2>
              <p style={{ fontSize: '0.9rem', color: '#666', marginTop: '-10px', marginBottom: '20px' }}>
                업로드한 문서를 기반으로 AI 퀴즈를 풀어보세요
              </p>

              {/* 퀴즈 설정 */}
              {!isQuizMode && (
                <div className="quiz-settings-section">
                  <div className="setting-group">
                    <label>파일 선택:</label>
                    <select
                      value={selectedFileUri}
                      onChange={(e) => setSelectedFileUri(e.target.value)}
                      className="setting-select"
                      disabled={pdfQuizLoading}
                    >
                      <option value="">파일을 선택하세요</option>
                      {uploadedFiles.map((file, index) => (
                        <option key={index} value={file.name}>
                          {file.display_name}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="setting-group">
                    <label>난이도:</label>
                    <select
                      value={difficulty}
                      onChange={(e) => setDifficulty(e.target.value)}
                      className="setting-select"
                      disabled={pdfQuizLoading}
                    >
                      <option value="easy">쉬움</option>
                      <option value="medium">보통</option>
                      <option value="hard">어려움</option>
                    </select>
                  </div>

                  <button
                    onClick={handleGenerateQuizFromFile}
                    disabled={pdfQuizLoading || !selectedFileUri || uploadedFiles.length === 0}
                    className="action-btn primary"
                  >
                    {pdfQuizLoading ? '퀴즈 생성 중...' : '퀴즈 시작'}
                  </button>

                  {uploadedFiles.length === 0 && (
                    <p className="empty-message">
                      홈 화면에서 PDF 파일을 먼저 업로드해주세요.
                    </p>
                  )}

                  <p style={{ fontSize: '0.85rem', color: '#8b5cf6', marginTop: '1rem', textAlign: 'center' }}>
                    💡 답을 제출하면 자동으로 다음 문제가 생성됩니다
                  </p>
                </div>
              )}

              {/* 채팅 섹션 */}
              <div className="chat-section">
                <div className="chat-history">
                  {pdfQuizMessages.map((msg) => (
                    <div
                      key={msg.id}
                      className={`chat-message ${msg.role}`}
                    >
                      <div className="message-label">
                        {msg.role === 'user' ? '👤 사용자' : '🤖 AI'}
                      </div>
                      <div className="message-content">
                        <pre style={{ whiteSpace: 'pre-wrap', wordWrap: 'break-word', margin: 0, fontFamily: 'inherit' }}>
                          {msg.content}
                        </pre>
                      </div>
                    </div>
                  ))}
                  {pdfQuizMessages.length === 1 && !isQuizMode && (
                    <p className="empty-message">
                      파일을 선택하고 퀴즈를 시작해주세요!
                    </p>
                  )}
                  {pdfQuizLoading && (
                    <div className="chat-message assistant">
                      <div className="message-label">🤖 AI</div>
                      <div className="message-content">
                        <span style={{ color: '#8b5cf6' }}>생각하는 중...</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* 입력창 */}
                {isQuizMode && (
                  <div className="chat-input">
                    <textarea
                      value={pdfQuizInput}
                      onChange={(e) => setPdfQuizInput(e.target.value)}
                      onKeyPress={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          handlePdfQuizSendMessage();
                        }
                      }}
                      placeholder="정답을 입력하세요..."
                      rows={3}
                      className="chat-textarea"
                      disabled={pdfQuizLoading}
                    />
                    <button
                      onClick={handlePdfQuizSendMessage}
                      disabled={pdfQuizLoading || !pdfQuizInput.trim()}
                      className="send-btn"
                    >
                      {pdfQuizLoading ? '전송 중...' : '전송'}
                    </button>
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
