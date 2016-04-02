// ==UserScript==
// @name        chaptcha
// @namespace   https://2chk.hk/chaptcha
// @description Automatically enter captcha on 2ch.hk using chaptcha.py backend
// @downloadURL https://raw.githubusercontent.com/Kagami/chaptcha/master/chaptcha.user.js
// @updateURL   https://raw.githubusercontent.com/Kagami/chaptcha/master/chaptcha.user.js
// @include     https://2ch.hk/*
// @version     0.0.1
// @grant       none
// ==/UserScript==

// TODO: Some better way to store that setting?
var OCR_BACKEND_URL = localStorage.getItem("OCR_BACKEND_URL") ||
                      "http://127.0.0.1:28228";

function getImageData(img) {
  return new Promise(function(resolve, reject) {
    img.onload = function() {
      var canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;
      var context = canvas.getContext("2d");
      context.drawImage(img, 0, 0);
      canvas.toBlob(resolve);
    };
    img.onerror = reject;
  });
}

function ocr(data) {
  return new Promise(function(resolve, reject) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", OCR_BACKEND_URL + "/ocr", true);
    xhr.onload = function() {
      if (this.status >= 200 && this.status < 400) {
        resolve(this.responseText);
      } else {
        reject(new Error(this.responseText));
      }
    };
    xhr.onerror = reject;
    var form = new FormData();
    form.append("file", data);
    xhr.send(form);
  });
}

function setAnswer(answer) {
  document.getElementById("captcha-value").value = answer;
  document.getElementById("qr-captcha-value").value = answer;
}

document.addEventListener("DOMContentLoaded", function() {
  var container = document.getElementById("captcha-widget-main");
  if (!container) return;
  var observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      Array.prototype.filter.call(mutation.addedNodes, function(node) {
        return node.tagName === "IMG";
      }).forEach(function(img) {
        setAnswer("");
        getImageData(img).then(ocr).then(function(answer) {
          setAnswer(answer);
        }, function() {
          // Indicate error.
          setAnswer("000000");
        });
      });
    });
  });
  // Captcha updates synchronously in all places so it's enough to
  // observe only single one.
  observer.observe(container, {childList: true});
}, false);
