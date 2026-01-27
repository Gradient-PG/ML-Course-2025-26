import { useState } from 'react';

export default function App() {
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [imageUrl, setImageUrl] = useState(null);

  const handleUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setImageUrl(URL.createObjectURL(file));

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${import.meta.env.VITE_BACKEND_URL}/predict`, {
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

      {imageUrl && (
        <div style={{ marginTop: '20px' }}>
          <h3>Uploaded Image:</h3>
          <img src={imageUrl} alt="Uploaded preview" style={{ maxWidth: '300px', maxHeight: '300px', border: '1px solid #ccc' }} />
        </div>
      )}

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