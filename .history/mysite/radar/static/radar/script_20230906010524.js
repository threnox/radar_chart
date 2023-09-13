'use strict'

const btn = document.getElementById('btn');
const frame = document.getElementById('frame');
const error = document.getElementsByClassName('errorlist')
const reset = document.getElementById('reset')

btn.addEventListener('click', function() {

    frame.style.display = 'block';

	    

}, false);

if (error.style.display = 'block') {
    reset.style.display = 'block';
};
// window.onload = function() {
// 	frame.style.display = 'none';
// 	}

// document.getElementById("btn").addEventListener("click", function() {
//     console.log('ボタンをクリック');
//     const loader = document.getElementById('frame');
//         var req = new XMLHttpRequest();
//         console.log('ロード中');
//         req.onreadystatechange = function() {
//             if (req.readyState == 1) {
//                 loader.style.display = "block";
//                 console.log('通信開始');
//                 if (req.status == 4) {
//                     console.log('通信完了');
//                     loader.style.display = "none";
//                 }
//             }else{
//                 loader.style.display = "block";
//                 console.log('通信中');
//             }
//             }
// });
