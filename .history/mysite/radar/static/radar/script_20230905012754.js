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
    const loader = document.querySelector("#frame");
        var req = new XMLHttpRequest();
        console.log('ロード中');
        req.onreadystatechange = function() {
            if (req.readyState == 3) {
                loader.style.visibility = "visible";
                console.log('通信開始');
                if (req.status == 200) {
                    console.log('通信完了');
                    loader.style.visibility = "hidden";
                }
            }else{
                loader.style.visibility = "visible";
                console.log('通信中');
            }
            }
});
