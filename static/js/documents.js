const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");

uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    uploadStatus.className = "status";
    uploadStatus.textContent = "Uploading and indexing...";

    try {
        const res = await fetch("/upload-doc", {
            method: "POST",
            body: formData,
        });
        const data = await res.json();

        if (!res.ok) {
            uploadStatus.className = "status error";
            uploadStatus.textContent = data.detail || "Upload failed.";
            return;
        }

        uploadStatus.className = "status success";
        uploadStatus.textContent = data.message || "Uploaded.";
        // Reload to refresh the document list table rendered by Jinja.
        setTimeout(() => window.location.reload(), 600);
    } catch (err) {
        uploadStatus.className = "status error";
        uploadStatus.textContent = `Network error: ${err.message}`;
    }
});

// Delete buttons (event delegation)
document.querySelectorAll(".delete-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
        const fileId = parseInt(btn.dataset.fileId, 10);
        if (!confirm(`Delete document #${fileId}?`)) return;

        try {
            const res = await fetch("/delete-doc", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ file_id: fileId }),
            });
            const data = await res.json();

            if (data.error) {
                alert(data.error);
                return;
            }

            // Remove row from the table without reloading.
            const row = document.querySelector(`tr[data-file-id="${fileId}"]`);
            if (row) row.remove();
        } catch (err) {
            alert(`Network error: ${err.message}`);
        }
    });
});
