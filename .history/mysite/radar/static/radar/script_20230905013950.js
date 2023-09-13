'use strict'

const btn = document.getElementById('btn');
const frame = document.getElementById('frame');

// btn.addEventListener('click', function() {

// 	frame.style.display = 'block';

// }, false);

// window.onload = function() {
// 	frame.style.display = 'none';
// 	}

document.getElementById("btn").addEventListener("click", function() {
    console.log('ボタンをクリック');
    const loader = document.getElementById('frame');
        var req = new XMLHttpRequest();
        console.log('ロード中');
        req.onreadystatechange = function() {
            if (req.readyState == 0) {
                loader.style.display = "block";
                console.log('通信開始');
                if (req.status == 4) {
                    console.log('通信完了');
                    loader.style.display = "none";
                }
            }else{
                loader.style.display = "block";
                console.log('通信中');
            }
            }
});
