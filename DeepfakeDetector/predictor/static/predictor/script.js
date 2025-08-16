function clearInputs() {
  document.getElementById('fileInput').value = '';
  const resultDiv = document.getElementById('result');
  resultDiv.innerHTML = '';
  resultDiv.className = '';
}

function showInfo() {
  alert('This platform uses AI to detect deepfake content in audio, images, and video files with high precision.');
}

document.getElementById("uploadForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const resultDiv = document.getElementById('result');
  const fileInput = document.getElementById('fileInput');
  const submitButton = e.target.querySelector('button[type="submit"]');

  if (fileInput.files.length === 0) {
    resultDiv.innerHTML = '<p class="error">Please select a file to upload.</p>';
    return;
  }

  resultDiv.innerHTML = '<p class="loading">Analyzing... Please wait.</p>';
  submitButton.disabled = true;

  const formData = new FormData();
  formData.append("media", fileInput.files[0]);
  const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;

  try {
    const response = await fetch("/predictor/detect/", {
      method: "POST",
      headers: {
        'X-CSRFToken': csrfToken,
      },
      body: formData,
    });

    const result = await response.json();

    resultDiv.className = result.class;
    resultDiv.innerHTML = `
      <h4>${result.text}</h4>
      ${result.Accuracy ? `<p class="confidence">Accuracy: <strong>${result.Accuracy}</strong></p>` : ''}
    `;

  } catch (error) {
    resultDiv.className = 'error';
    resultDiv.innerHTML = '<h4>An error occurred.</h4><p>Could not connect to the server.</p>';
    console.error("Fetch error:", error);
  } finally {
    submitButton.disabled = false;
  }
});
