function [] = findSounds( myNotes )
% This function gets calculated positions of notes and returns the
% sinusoidals which gives the simple musical equivalent of the sinusoidals
    mySounds = [880.0 784.0 698.5 659.3 587.3 523.3 493.9 440.0 392.0 349.2 329.6 293.7 261.6];

    Fs=8000;
    Ts=1/Fs;
    A = [];

    for i=1:length(myNotes)
        t=[0:Ts:0.3*myNotes(i,2)];
        A = [A sin(2*pi*mySounds(myNotes(i,1))*t)];
    end

    sound(A);
end

