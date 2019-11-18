#!/bin/zsh

if [ $1 = "mac" ]
then
    wget https://stockfishchess.org/files/stockfish-10-mac.zip &&\
        unzip stockfish-10-mac.zip &&\
        mv stockfish-10-mac/Mac/stockfish-10-64 . &&\
        rm -rf stockfish-10-mac stockfish-10-mac.zip __MACOSX
fi

if [ $1 = "linux" ]
then
    wget https://stockfishchess.org/files/stockfish-10-linux.zip &&\
        unzip stockfish-10-linux.zip &&\
        mv stockfish-10-linux/Linux/stockfish_10_x64 ./stockfish-10-64 &&\
        rm -rf stockfish-10-linux* &&\
        chmod +x stockfish-10-64
fi
