import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Home.css';

function Home() {
  const navigate = useNavigate();

  return (
    <div className="home-container">
      <div className="home-content">
        <h1 className="site-title">Study Helper</h1>
        <p className="site-subtitle">국가기술자격 시험 준비를 도와드립니다</p>

        <div className="feature-cards">
          <div
            className="feature-card"
            onClick={() => navigate('/chatbot')}
          >
            <div className="card-icon">🤖</div>
            <h2 className="card-title">AI 챗봇</h2>
            <p className="card-description">
              학습 관련 질문에 대해 AI가 친절하게 답변해드립니다
            </p>
          </div>

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
        </div>
      </div>
    </div>
  );
}

export default Home;
