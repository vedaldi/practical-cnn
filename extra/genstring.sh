#!/bin/bash
# file: genfonts.sh
# brief: generate font glyphs for exercise4.m
# author: Andrea Vedaldi
#
# This script uses ImageMagik convert to generate images of glyphs
# starting from Google's font collection. It requires ImageMagik as
# well as mercurial (to download Google's fonts).
#
# Run this script from the root folder of the practical.
#
# Do not forget to pack the font using the extra/packFonts.m script.


googleFontDir="extra/googlefontdirectory"

convert \
    -size 576x32 \
    -background white -fill black \
    -font "$googleFontDir/"ofl/lato/Lato-Regular.ttf \
    -pointsize 26 \
    -gravity center \
    label:"the rat the cat the dog chased killed ate the malt" \
    "data/sentence-lato.png"
