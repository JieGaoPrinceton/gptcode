% 生成示例 IQ 数据
t = 0:0.01:1; % 时间向量
f = 5; % 信号频率
I = cos(2 * pi * f * t); % 同相分量
Q = sin(2 * pi * f * t); % 正交分量
IQ = I + 1i * Q; % Complex form of IQ data


% 显示 IQ 数据
disp('IQ Data:');
disp(IQ);

% 绘制 IQ 数据
figure;
subplot(2, 1, 1);
plot(t, real(IQ));
title('In-phase Component (I)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
subplot(2, 1, 2);
plot(t, imag(IQ));
title('Quadrature Component (Q)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
ylabel('Amplitude');