import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { ThemeProvider } from './providers/ThemeProvider';
import { Layout } from './components/layout/Layout';
import { Dashboard } from './pages/Dashboard';
import { Transactions } from './pages/Transactions';
import { Calibration } from './pages/Calibration';
import { Investigation } from './pages/Investigation';
import ManualReview from './pages/ManualReview';
import Monitoring from './pages/Monitoring';
import { Reports } from './pages/Reports';
import { Alerts } from './pages/Alerts';
import { Datasets } from './pages/Datasets';
import HardRules from './pages/HardRules';
import VipList from './pages/VipList';
import HotList from './pages/HotList';
import Metrics from './pages/Metrics';
import { Audit } from './pages/Audit';
import { Settings } from './pages/Settings';
import './App.css';

function AppContent() {
  const location = useLocation();

  return (
    <Layout currentPath={location.pathname}>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/transactions" element={<Transactions />} />
        <Route path="/calibration" element={<Calibration />} />
        <Route path="/investigation" element={<Investigation />} />
        <Route path="/manual-review" element={<ManualReview />} />
        <Route path="/monitoring" element={<Monitoring />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/metrics" element={<Metrics />} />
        <Route path="/alerts" element={<Alerts />} />
        <Route path="/datasets" element={<Datasets />} />
        <Route path="/hard-rules" element={<HardRules />} />
        <Route path="/vip-list" element={<VipList />} />
        <Route path="/hot-list" element={<HotList />} />
        <Route path="/audit" element={<Audit />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="*" element={<div className="p-8 text-center">Página não encontrada</div>} />
      </Routes>
    </Layout>
  );
}

function App() {
  return (
    <ThemeProvider>
      <Router>
        <AppContent />
      </Router>
    </ThemeProvider>
  );
}

export default App;

