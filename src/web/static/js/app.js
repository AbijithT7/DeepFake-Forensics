// DeepFake Forensics - Client Interactions
document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("upload-dropzone");
  const fileInput = document.getElementById("file-input");
  const previewContainer = document.getElementById("preview-container");
  const previewImg = document.getElementById("preview-img");
  const uploadForm = document.getElementById("upload-form");
  const submitBtn = document.getElementById("submit-btn");

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelect(fileInput.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
      }
    });
  }

  function handleFileSelect(file) {
    if (!file.type.match("image.*")) {
      alert("Please select a valid image file (JPEG, PNG).");
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      if (previewImg && previewContainer) {
        previewImg.src = e.target.result;
        previewContainer.style.display = "block";
      }
    };
    reader.readAsDataURL(file);
  }

  // Handle sample image quick select buttons
  const sampleBtns = document.querySelectorAll(".btn-sample-select");
  sampleBtns.forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const samplePath = e.currentTarget.getAttribute("data-sample");
      const sampleInput = document.getElementById("sample-input");
      if (sampleInput && uploadForm) {
        sampleInput.value = samplePath;
        if (submitBtn) {
          submitBtn.innerHTML = "<span>Analyzing Sample...</span>";
          submitBtn.disabled = true;
        }
        uploadForm.submit();
      }
    });
  });

  if (uploadForm && submitBtn) {
    uploadForm.addEventListener("submit", () => {
      submitBtn.innerHTML = "<span>Processing DeepFake Forensics...</span>";
      submitBtn.disabled = true;
    });
  }
});
