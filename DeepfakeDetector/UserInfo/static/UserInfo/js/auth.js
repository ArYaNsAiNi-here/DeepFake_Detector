document.addEventListener("DOMContentLoaded", () => {
  const inputs = document.querySelectorAll(".auth-form input");
  inputs.forEach(input => {
    input.addEventListener("focus", () => {
      input.style.backgroundColor = "#f0f9ff";
    });
    input.addEventListener("blur", () => {
      input.style.backgroundColor = "white";
    });
  });
});
