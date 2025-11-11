import React from 'react';
import { Link } from 'react-router-dom';
import './Navigation.css';

function Navigation() {
  return (
    <nav className="navigation">
      <div className="nav-container">
        <Link to="/" className="nav-brand">
          Q-Pass
        </Link>
        <ul className="nav-menu">
          <li className="nav-item">
            <Link to="/study-plan" className="nav-link">학습 계획</Link>
          </li>
          <li className="nav-item">
            <Link to="/document-analyzer" className="nav-link">자료 분석</Link>
          </li>
        </ul>
      </div>
    </nav>
  );
}

export default Navigation;
