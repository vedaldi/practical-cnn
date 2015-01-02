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
fonts=$(echo "$googleFontDir/ofl/*/*.ttf")
chars="a b c d e f g h i j k l m n o p q r s t u v w x y z"

hg clone https://code.google.com/p/googlefontdirectory/ \
    -r 3859 \
    "$googleFontDir"

for i in $fonts
do
    fontName=$(basename "$i" .ttf)
    fondDestDir="extra/fonts/$fontName"
    echo "$fontName"
    mkdir -p "$fondDestDir"
    for c in $chars
    do
        convert \
            -size 32x32 \
            -background white -fill black \
            -font "$i" \
            -pointsize 26 \
            -gravity center \
            label:$c \
            "$fontDestDir/$c.png"
    done
done
