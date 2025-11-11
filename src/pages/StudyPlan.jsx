import React, { useState, useEffect, useRef } from 'react';
import { getQualificationList } from '../services/qnetApi';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import './StudyPlan.css';

function StudyPlan() {
  const [loading, setLoading] = useState(false);
  const [qualificationList, setQualificationList] = useState([]);
  const [loadingQualifications, setLoadingQualifications] = useState(false);

  // API 중복 호출 방지를 위한 ref
  const hasLoadedQualifications = useRef(false);

  // 검색 및 선택 관련 state
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSubject, setSelectedSubject] = useState(null);

  // 직무분야 필터 관련 state
  const [selectedObligFld, setSelectedObligFld] = useState('');
  const [selectedMdObligFld, setSelectedMdObligFld] = useState('');

  // 시험 일정 관련 state
  const [examSchedules, setExamSchedules] = useState([]);
  const [loadingSchedules, setLoadingSchedules] = useState(false);
  const [selectedSchedule, setSelectedSchedule] = useState(null);

  // 공부 시작 날짜
  const [startDate, setStartDate] = useState('');

  // 학습 계획 관련 state
  const [studyPlan, setStudyPlan] = useState(null);
  const [error, setError] = useState(null);

  // 탭 관리 state
  const [activeTab, setActiveTab] = useState(1);

  // 컴포넌트 마운트 시 종목 목록 로드 (중복 호출 방지)
  useEffect(() => {
    if (!hasLoadedQualifications.current) {
      hasLoadedQualifications.current = true;
      loadQualifications();
    }
  }, []);

  // 종목 선택 시 시험 일정 로드
  useEffect(() => {
    if (selectedSubject) {
      console.log('🎯 useEffect triggered for subject:', selectedSubject);
      loadExamSchedules();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSubject]);

  const loadQualifications = async () => {
    setLoadingQualifications(true);
    try {
      const response = await getQualificationList();

      if (response.success) {
        let items = [];

        if (response.data?.response?.body?.items?.item) {
          const rawItems = response.data.response.body.items.item;
          items = Array.isArray(rawItems) ? rawItems : [rawItems];
        } else if (response.data?.response?.body?.item) {
          const rawItems = response.data.response.body.item;
          items = Array.isArray(rawItems) ? rawItems : [rawItems];
        }

        setQualificationList(items);
      } else {
        console.error('Failed to load qualifications:', response.error);
        setError('종목 목록을 불러오는데 실패했습니다: ' + response.error);
      }
    } catch (err) {
      console.error('Error loading qualifications:', err);
      setError('오류가 발생했습니다: ' + err.message);
    } finally {
      setLoadingQualifications(false);
    }
  };

  // 시험 일정 로드
  const loadExamSchedules = async () => {
    if (!selectedSubject) return;

    console.log('🔍 Loading exam schedules for:', selectedSubject.name, selectedSubject.code);

    setLoadingSchedules(true);
    setExamSchedules([]);
    setSelectedSchedule(null);
    setError(null);

    try {
      const currentYear = new Date().getFullYear();
      const url = `http://localhost:3001/api/qnet/jm-list?jmCd=${selectedSubject.code}&implYy=${currentYear}`;
      console.log('📡 Fetching:', url);

      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const xmlText = await response.text();

      // XML 파싱
      const parser = new DOMParser();
      const xmlDoc = parser.parseFromString(xmlText, 'text/xml');

      // 파싱 에러 체크
      const parserError = xmlDoc.getElementsByTagName('parsererror');
      if (parserError.length > 0) {
        throw new Error('XML 파싱 실패');
      }

      const items = xmlDoc.getElementsByTagName('item');
      console.log('📊 Found items:', items.length);

      const schedules = [];
      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        const schedule = {
          description: getXMLValue(item, 'description') || getXMLValue(item, 'implplannm') || `${i + 1}회차`,
          docRegStartDt: getXMLValue(item, 'docRegStartDt') || getXMLValue(item, 'docregstartdt'),
          docRegEndDt: getXMLValue(item, 'docRegEndDt') || getXMLValue(item, 'docregenddt'),
          docExamDt: getXMLValue(item, 'docExamDt') || getXMLValue(item, 'docexamdt') || getXMLValue(item, 'docexamstartdt'),
          docPassDt: getXMLValue(item, 'docPassDt') || getXMLValue(item, 'docpassdt'),
          pracRegStartDt: getXMLValue(item, 'pracRegStartDt') || getXMLValue(item, 'pracregstartdt'),
          pracRegEndDt: getXMLValue(item, 'pracRegEndDt') || getXMLValue(item, 'pracregenddt'),
          pracExamStartDt: getXMLValue(item, 'pracExamStartDt') || getXMLValue(item, 'pracexamstartdt'),
          pracExamEndDt: getXMLValue(item, 'pracExamEndDt') || getXMLValue(item, 'pracexamenddt'),
          pracPassDt: getXMLValue(item, 'pracPassDt') || getXMLValue(item, 'pracpassdt') || getXMLValue(item, 'pracpassstartdt'),
        };

        schedules.push(schedule);
      }

      console.log('✅ Schedules loaded:', schedules.length);
      setExamSchedules(schedules);

      // 첫 번째 일정 자동 선택
      if (schedules.length > 0) {
        setSelectedSchedule(schedules[0]);
        console.log('✅ Auto-selected first schedule');
      }

    } catch (err) {
      console.error('❌ Error loading exam schedules:', err);
      setError('시험 일정을 불러오는데 실패했습니다: ' + err.message);
    } finally {
      setLoadingSchedules(false);
    }
  };

  // XML 값 추출 헬퍼 함수
  const getXMLValue = (item, tagName) => {
    const element = item.getElementsByTagName(tagName)[0];
    return element ? element.textContent : '';
  };

  // 날짜 포맷 함수
  const formatDate = (dateStr) => {
    if (!dateStr || dateStr.length !== 8) return dateStr;
    const year = dateStr.substring(0, 4);
    const month = dateStr.substring(4, 6);
    const day = dateStr.substring(6, 8);
    return `${year}년 ${month}월 ${day}일`;
  };

  // 대직무분야 목록 추출
  const obligFldList = React.useMemo(() => {
    const uniqueObligFlds = new Map();
    qualificationList.forEach(item => {
      const code = item.obligfldcd || item.obligFldCd;
      const name = item.obligfldnm || item.obligFldNm;
      if (code && name && typeof name === 'string' && name.trim() && !uniqueObligFlds.has(code)) {
        uniqueObligFlds.set(code, name.trim());
      }
    });
    return Array.from(uniqueObligFlds, ([code, name]) => ({ code, name }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [qualificationList]);

  // 중직무분야 목록 추출
  const mdObligFldList = React.useMemo(() => {
    if (!selectedObligFld) return [];

    const uniqueMdObligFlds = new Map();
    qualificationList
      .filter(item => {
        const obligCode = item.obligfldcd || item.obligFldCd;
        return obligCode === selectedObligFld;
      })
      .forEach(item => {
        const code = item.mdobligfldcd || item.mdObligFldCd;
        const name = item.mdobligfldnm || item.mdObligFldNm;
        if (code && name && typeof name === 'string' && name.trim() && !uniqueMdObligFlds.has(code)) {
          uniqueMdObligFlds.set(code, name.trim());
        }
      });
    return Array.from(uniqueMdObligFlds, ([code, name]) => ({ code, name }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [qualificationList, selectedObligFld]);

  // 종목 필터링
  const filteredQualifications = qualificationList.filter(item => {
    if (selectedObligFld) {
      const obligCode = item.obligfldcd || item.obligFldCd;
      if (obligCode !== selectedObligFld) return false;
    }

    if (selectedMdObligFld) {
      const mdObligCode = item.mdobligfldcd || item.mdObligFldCd;
      if (mdObligCode !== selectedMdObligFld) return false;
    }

    if (searchTerm.trim()) {
      const searchLower = searchTerm.toLowerCase();
      const jmNm = (item.jmfldnm || item.jmNm || '').toLowerCase();
      const jmCd = (item.jmcd || item.jmCd || '').toLowerCase();
      return jmNm.includes(searchLower) || jmCd.includes(searchLower);
    }

    return true;
  });

  // 종목 선택 핸들러
  const handleSubjectSelect = (item) => {
    const jmCd = item.jmcd || item.jmCd;
    const jmNm = item.jmfldnm || item.jmNm;

    setSelectedSubject({
      code: jmCd,
      name: jmNm,
      item: item
    });
    setSearchTerm(jmNm);
    setSelectedSchedule(null);
    setStartDate('');
    setStudyPlan(null);
    setError(null);

    // 2단계 탭으로 자동 이동
    setActiveTab(2);
  };

  // 학습 계획 생성
  const generateStudyPlan = async () => {
    if (!selectedSubject) {
      setError('종목을 선택해주세요.');
      return;
    }

    if (!selectedSchedule) {
      setError('시험 일정을 선택해주세요.');
      return;
    }

    if (!startDate) {
      setError('공부 시작 날짜를 선택해주세요.');
      return;
    }

    setLoading(true);
    setError(null);
    setStudyPlan(null);

    try {
      // OpenAI API 호출
      const response = await fetch('http://localhost:3001/api/openai/generate-study-plan', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          subject: selectedSubject.name,
          exam_schedule: selectedSchedule,
          start_date: startDate
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate study plan');
      }

      const data = await response.json();
      setStudyPlan(data);

    } catch (err) {
      console.error('Error generating study plan:', err);
      setError('학습 계획 생성 중 오류가 발생했습니다: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // 초기화
  const handleReset = () => {
    setSelectedObligFld('');
    setSelectedMdObligFld('');
    setSelectedSubject(null);
    setSearchTerm('');
    setExamSchedules([]);
    setSelectedSchedule(null);
    setStartDate('');
    setStudyPlan(null);
    setError(null);
    setActiveTab(1); // 1단계 탭으로 돌아가기
  };

  // 시험 일정 선택 핸들러
  const handleScheduleSelect = (schedule) => {
    setSelectedSchedule(schedule);
    setActiveTab(3); // 3단계 탭으로 이동
  };

  // 날짜 선택 핸들러
  const handleDateSelect = (date) => {
    setStartDate(date);
  };

  return (
    <div className="page-container">
      <h1>AI 학습 계획 생성기</h1>
      <p>응시하고 싶은 종목을 선택하면 AI가 맞춤 학습 계획을 생성해드립니다</p>

      <div className="study-plan-container">
        {/* 탭 네비게이션과 선택 내역 */}
        <div className="navigation-with-summary">
          {/* 탭 네비게이션 */}
          <div className="tab-navigation">
            <div
              className={`tab-item ${activeTab === 1 ? 'active' : ''} ${selectedSubject ? 'completed' : ''}`}
              onClick={() => setActiveTab(1)}
            >
              <div className="tab-number">1</div>
              <div className="tab-label">종목 선택</div>
            </div>
            <div className="tab-divider"></div>
            <div
              className={`tab-item ${activeTab === 2 ? 'active' : ''} ${selectedSchedule ? 'completed' : ''} ${!selectedSubject ? 'disabled' : ''}`}
              onClick={() => selectedSubject && setActiveTab(2)}
            >
              <div className="tab-number">2</div>
              <div className="tab-label">시험 일정</div>
            </div>
            <div className="tab-divider"></div>
            <div
              className={`tab-item ${activeTab === 3 ? 'active' : ''} ${startDate ? 'completed' : ''} ${!selectedSchedule ? 'disabled' : ''}`}
              onClick={() => selectedSchedule && setActiveTab(3)}
            >
              <div className="tab-number">3</div>
              <div className="tab-label">시작 날짜</div>
            </div>
            <div className="tab-divider"></div>
            <div
              className={`tab-item ${activeTab === 4 ? 'active' : ''} ${studyPlan ? 'completed' : ''} ${!startDate ? 'disabled' : ''}`}
              onClick={() => startDate && setActiveTab(4)}
            >
              <div className="tab-number">4</div>
              <div className="tab-label">학습 계획 생성</div>
            </div>
          </div>

          {/* 선택 내역 요약 (종목 선택 후 표시) */}
          {selectedSubject && (
            <div className="selection-summary-right">
              <h3>📋 선택 내역</h3>
              <div className="summary-items">
                <div className="summary-item">
                  <span className="summary-label">선택한 종목:</span>
                  <span className="summary-value">{selectedSubject.name}</span>
                </div>
                {selectedSchedule && (
                  <>
                    <div className="summary-item">
                      <span className="summary-label">선택한 시험 일정:</span>
                      <span className="summary-value">{selectedSchedule.description}</span>
                    </div>
                    {selectedSchedule.docExamDt && (
                      <div className="summary-item detail">
                        <span className="summary-label">📖 필기시험일:</span>
                        <span className="summary-value">{formatDate(selectedSchedule.docExamDt)}</span>
                      </div>
                    )}
                    {selectedSchedule.pracExamStartDt && (
                      <div className="summary-item detail">
                        <span className="summary-label">🔧 실기시험일:</span>
                        <span className="summary-value">{formatDate(selectedSchedule.pracExamStartDt)}</span>
                      </div>
                    )}
                  </>
                )}
                {startDate && (
                  <div className="summary-item">
                    <span className="summary-label">공부 시작 날짜:</span>
                    <span className="summary-value">{new Date(startDate).toLocaleDateString('ko-KR', {
                      year: 'numeric',
                      month: 'long',
                      day: 'numeric',
                      weekday: 'long'
                    })}</span>
                  </div>
                )}
              </div>
              <button
                className="change-subject-button"
                onClick={handleReset}
                style={{ marginTop: '1rem', width: '100%' }}
              >
                종목 변경
              </button>
            </div>
          )}
        </div>

        {/* 탭 컨텐츠 영역 */}
        <div className="tab-content">
          {/* 1단계: 종목 선택 */}
          {activeTab === 1 && (
            <div className="tab-panel">
              <div className="selection-section">
                <h2>종목 선택</h2>

                {loadingQualifications ? (
                  <div className="loading-message">
                    <p>종목 목록을 불러오는 중...</p>
                  </div>
                ) : (
            <>
              {/* 전역 검색 박스 - 아무것도 선택하지 않았을 때만 표시 */}
              {!selectedSubject && !selectedObligFld && !selectedMdObligFld && (
                <div className="global-search-section">
                  <div className="search-box">
                    <input
                      type="text"
                      placeholder="🔍 종목명으로 직접 검색하기... (예: 정보처리기사)"
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="search-input"
                    />
                  </div>
                  {searchTerm.trim() && filteredQualifications.length > 0 && (
                    <div className="search-results">
                      <p className="search-result-count">
                        🎯 {filteredQualifications.length}개의 종목이 검색되었습니다
                      </p>
                      <div className="subject-grid">
                        {filteredQualifications.slice(0, 20).map((item, index) => {
                          const jmCode = item.jmcd || item.jmCd;
                          const jmName = item.jmfldnm || item.jmNm;
                          return (
                            <div
                              key={`search-jm-${String(jmCode)}-${index}`}
                              className="subject-card"
                              onClick={() => handleSubjectSelect(item)}
                            >
                              <div className="subject-name">{jmName}</div>
                            </div>
                          );
                        })}
                      </div>
                      {filteredQualifications.length > 20 && (
                        <p className="more-results-hint">
                          💡 더 많은 결과가 있습니다. 검색어를 더 구체적으로 입력해보세요.
                        </p>
                      )}
                    </div>
                  )}
                  {searchTerm.trim() && filteredQualifications.length === 0 && (
                    <div className="no-results">
                      <p>❌ 검색 결과가 없습니다. 다른 검색어를 시도해보세요.</p>
                    </div>
                  )}
                  <div className="divider">
                    <span>또는 카테고리로 찾기</span>
                  </div>
                </div>
              )}

              {/* 카테고리 기반 선택: 3단계 목록 뷰 */}
              {!searchTerm.trim() && (
                <div className="list-selection-container">
                  {/* 1단계: 대직무분야 목록 */}
                  <div className="list-panel">
                    <h3>대직무분야</h3>
                    <div className="list-items">
                      {obligFldList.map((item, index) => (
                        <div
                          key={`oblig-${String(item.code)}-${index}`}
                          className={`list-item ${selectedObligFld === item.code ? 'selected' : ''}`}
                          onClick={() => {
                            setSelectedObligFld(item.code);
                            setSelectedMdObligFld('');
                            setSelectedSubject(null);
                            setSearchTerm('');
                          }}
                        >
                          <span className="list-icon">📁</span>
                          <span className="list-name">{item.name}</span>
                          <span className="list-arrow">›</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* 2단계: 중직무분야 목록 */}
                  {selectedObligFld && (
                    <div className="list-panel">
                      <h3>중직무분야</h3>
                      <div className="list-items">
                        {mdObligFldList.map((item, index) => (
                          <div
                            key={`mdoblig-${String(item.code)}-${index}`}
                            className={`list-item ${selectedMdObligFld === item.code ? 'selected' : ''}`}
                            onClick={() => {
                              setSelectedMdObligFld(item.code);
                              setSelectedSubject(null);
                              setSearchTerm('');
                            }}
                          >
                            <span className="list-icon">📂</span>
                            <span className="list-name">{item.name}</span>
                            <span className="list-arrow">›</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* 3단계: 종목 목록 */}
                  {selectedMdObligFld && (
                    <div className="list-panel list-panel-wide">
                      <div className="panel-header">
                        <h3>종목 선택</h3>
                        <button className="reset-button-small" onClick={handleReset}>
                          ↺ 처음부터 다시
                        </button>
                      </div>

                      {/* 종목 내 검색 */}
                      <div className="search-box-inline">
                        <input
                          type="text"
                          placeholder="🔍 종목명 검색..."
                          value={searchTerm}
                          onChange={(e) => setSearchTerm(e.target.value)}
                          className="search-input-small"
                        />
                      </div>

                      <div className="list-items">
                        {filteredQualifications.map((item, index) => {
                          const jmCode = item.jmcd || item.jmCd;
                          const jmName = item.jmfldnm || item.jmNm;
                          return (
                            <div
                              key={`jm-${String(jmCode)}-${index}`}
                              className="list-item"
                              onClick={() => handleSubjectSelect(item)}
                            >
                              <span className="list-icon">📄</span>
                              <span className="list-name">{jmName}</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
              </div>
            </div>
          )}

          {/* 2단계: 시험 일정 선택 */}
          {activeTab === 2 && selectedSubject && (
            <div className="tab-panel">
              <div className="selection-section">
                <h2>시험 일정 선택</h2>

                {loadingSchedules ? (
                  <div className="loading-message">
                    <p>⏳ 시험 일정을 불러오는 중...</p>
                  </div>
                ) : error && !examSchedules.length ? (
                  <div className="error-message-box">
                    <h3>⚠️ 시험 일정을 불러올 수 없습니다</h3>
                    <p>{error}</p>
                    <button className="retry-button" onClick={loadExamSchedules}>
                      🔄 다시 시도
                    </button>
                  </div>
                ) : examSchedules.length === 0 ? (
                  <div className="info-message">
                    <p>📅 해당 종목의 시험 일정 정보가 아직 등록되지 않았습니다.</p>
                    <p className="info-sub">시험 일정이 공고되면 자동으로 업데이트됩니다.</p>
                  </div>
                ) : (
                  <div className="schedule-list">
                    {examSchedules.map((schedule, index) => (
                      <div
                        key={index}
                        className={`schedule-card ${selectedSchedule === schedule ? 'selected' : ''}`}
                        onClick={() => handleScheduleSelect(schedule)}
                  >
                    <h3>{schedule.description}</h3>
                    <div className="schedule-details">
                      {schedule.docRegStartDt && (
                        <p>📝 필기 원서접수: {formatDate(schedule.docRegStartDt)} ~ {formatDate(schedule.docRegEndDt)}</p>
                      )}
                      {schedule.docExamDt && (
                        <p>📖 필기시험: {formatDate(schedule.docExamDt)}</p>
                      )}
                      {schedule.docPassDt && (
                        <p>📋 필기 합격발표: {formatDate(schedule.docPassDt)}</p>
                      )}
                      {schedule.pracRegStartDt && (
                        <p>📝 실기 원서접수: {formatDate(schedule.pracRegStartDt)} ~ {formatDate(schedule.pracRegEndDt)}</p>
                      )}
                      {schedule.pracExamStartDt && (
                        <p>🔧 실기시험: {formatDate(schedule.pracExamStartDt)} ~ {formatDate(schedule.pracExamEndDt)}</p>
                      )}
                      {schedule.pracPassDt && (
                        <p>🎉 최종 합격발표: {formatDate(schedule.pracPassDt)}</p>
                      )}
                    </div>
                  </div>
                ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* 3단계: 공부 시작 날짜 선택 */}
          {activeTab === 3 && selectedSchedule && (
            <div className="tab-panel">
              <div className="selection-section">
                <h2>공부 시작 날짜 선택</h2>

                <div className="date-picker-section">
                  <label htmlFor="start-date">공부를 시작할 날짜를 선택하세요:</label>
                  <input
                    id="start-date"
                    type="date"
                    value={startDate}
                    onChange={(e) => handleDateSelect(e.target.value)}
                className="date-input"
              />
              {startDate && (
                <>
                  <p className="date-info">
                    선택된 날짜: <strong>{new Date(startDate).toLocaleDateString('ko-KR', {
                      year: 'numeric',
                      month: 'long',
                      day: 'numeric',
                      weekday: 'long'
                    })}</strong>
                  </p>
                  <button
                    className="next-button"
                    onClick={() => setActiveTab(4)}
                  >
                    다음 단계로 →
                  </button>
                </>
              )}
                </div>
              </div>
            </div>
          )}

          {/* 4단계: AI 학습 계획 생성 */}
          {activeTab === 4 && selectedSubject && selectedSchedule && startDate && (
            <div className="tab-panel">
              <div className="generate-section">
                <h2>AI 학습 계획 생성</h2>

                <button
                  className="generate-button"
                  onClick={generateStudyPlan}
                  disabled={loading}
                >
                  {loading ? '학습 계획 생성 중...' : 'AI 학습 계획 생성하기'}
                </button>
              </div>

              {/* 에러 메시지 */}
              {error && (
                <div className="error-section">
                  <h3>오류</h3>
                  <p>{error}</p>
                </div>
              )}

              {/* 학습 계획 결과 */}
              {studyPlan && (
                <div className="result-section">
                  <h2>✨ 맞춤 학습 계획</h2>
                  <div className="study-plan-content">
                    <h3>{studyPlan.subject}</h3>
                    <div className="plan-text markdown-content">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeRaw]}
                      >
                        {studyPlan.study_plan}
                      </ReactMarkdown>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default StudyPlan;
