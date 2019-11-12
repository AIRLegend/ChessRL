#!/bin/zsh

wget https://stockfishchess.org/files/stockfish-10-mac.zip &&\
    unzip stockfish-10-mac.zip &&\
    mv stockfish-10-mac/Mac/stockfish-10-64 . &&\
    rm -rf stockfish-10-mac stockfish-10-mac.zip __MACOSX
