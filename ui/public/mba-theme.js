const faviconUrl = "/public/favicon.png?v=7500cd6";
const icons = document.querySelectorAll('link[rel="icon"], link[rel="shortcut icon"]');

if (icons.length) {
  icons.forEach((icon) => {
    icon.href = faviconUrl;
    icon.type = "image/png";
  });
} else {
  const icon = document.createElement("link");
  icon.rel = "icon";
  icon.type = "image/png";
  icon.href = faviconUrl;
  document.head.appendChild(icon);
}
