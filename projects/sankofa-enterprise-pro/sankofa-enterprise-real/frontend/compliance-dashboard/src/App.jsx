import { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [complianceStatus, setComplianceStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchComplianceStatus = async () => {
      try {
        const response = await fetch('/api/v1/compliance/status', {
          headers: {
            'Authorization': `Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoyLCJ1c2VybmFtZSI6InRlc3RfYWRtaW4iLCJyb2xlIjoiYWRtaW4iLCJwZXJtaXNzaW9ucyI6WyJyZWFkX2FsbCIsIndyaXRlX2FsbCIsImRlbGV0ZV9hbGwiLCJtYW5hZ2VfdXNlcnMiLCJ2aWV3X2F1ZGl0Iiwic3lzdGVtX2NvbmZpZyIsImZyYXVkX2FuYWx5c2lzIl0sImlhdCI6MTc1ODQ3NjY3MywiZXhwIjoxNzU4NTA1NDczLCJ0eXBlIjoiYWNjZXNzIn0.F0eUe5M-yX2fOSTrUUgjQn7pQGI7Ue8nQKOqluw02kI`
          }
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        setComplianceStatus(data.data);
      } catch (error) {
        setError(error.message);
      }
    };

    fetchComplianceStatus();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Sankofa Compliance Dashboard</h1>
      </header>
      <main>
        {error && <p className="error">Erro ao carregar dados: {error}</p>}
        {complianceStatus ? (
          <div className="status-container">
            <h2>Status de Compliance</h2>
            <ul>
              {Object.entries(complianceStatus).map(([key, value]) => (
                <li key={key}>
                  <strong>{key.replace(/_/g, ' ').toUpperCase()}:</strong> {value}
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <p>Carregando status de compliance...</p>
        )}
      </main>
    </div>
  );
}

export default App;
