import React from 'react';
import { Link } from 'react-router-dom';
import './Navigation.css';

function Navigation() {
  return (
    <nav className="navigation">
      <div className="nav-container">
        <Link to="/" className="nav-brand">
          Study Helper
        </Link>
        <ul className="nav-menu">
          <li className="nav-item">
            <Link to="/chatbot" className="nav-link">AI 챗봇</Link>
          </li>
          <li className="nav-item">
            <Link to="/study-plan" className="nav-link">학습 계획</Link>
          </li>
        </ul>
      </div>
    </nav>
  );
}

export default Navigation;
