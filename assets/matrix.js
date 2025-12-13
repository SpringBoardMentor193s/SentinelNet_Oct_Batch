const canvas = document.createElement('canvas');
canvas.classList.add('matrix-bg');
document.body.appendChild(canvas);

const ctx = canvas.getContext('2d');
canvas.height = window.innerHeight;
canvas.width = window.innerWidth;

const letters = Array(256).join(1).split('');

function draw() {
    ctx.fillStyle = "rgba(0, 0, 0, 0.05)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.fillStyle = "#00ff9d";
    letters.map((y_pos, index) => {
        const text = String.fromCharCode(0x30A0 + Math.random() * 96);
        const x_pos = index * 15;

        ctx.fillText(text, x_pos, y_pos);

        letters[index] = (y_pos > canvas.height + Math.random() * 10000) ? 0 : y_pos + 15;
    });
}

setInterval(draw, 40);
