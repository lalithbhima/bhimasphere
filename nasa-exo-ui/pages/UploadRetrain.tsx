import React, { useState } from "react";

export default function UploadRetrain() {
  const [file, setFile] = useState<File | null>(null);

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    await fetch("http://127.0.0.1:7860/api/upload", {
      method: "POST",
      body: formData,
    });
    alert("Uploaded for retraining!");
  };

  return (
    <div>
      <h2>Upload & Retrain</h2>
      <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
      <button onClick={handleUpload}>Upload</button>
    </div>
  );
}
