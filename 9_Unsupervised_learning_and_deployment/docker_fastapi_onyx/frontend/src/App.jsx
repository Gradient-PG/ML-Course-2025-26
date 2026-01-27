import { useState } from 'react';

export default function App() {
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://localhost:8080/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setResponse(data);
    } catch (error) {
      console.error(error);
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>Photo Uploader</h1>
      <input type="file" accept="image/*" onChange={handleUpload} />

      {loading && <p>Uploading...</p>}

      {response && (
        <div style={{ marginTop: '20px' }}>
          <h3>Server Response:</h3>
          {/* Display the full JSON response */}
          <pre>{JSON.stringify(response[0], null, 2)}</pre>
        </div>
      )}
    </div>
  );
}