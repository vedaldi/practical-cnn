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
chars="a b c d e f g h i j k l m n o p q r s t u v w x y z"

# Download Google's fonts
if test ! -e "$googleFontDir"
then
    git clone https://github.com/google/fonts/ "$googleFontDir"
    (cd "$googleFontDir" ; git checkout bdc1940)
fi

# Only get the fonts that have latin-ext encoding
fontFamilies=$(grep -l latin-ext "$googleFontDir"/ofl/*/METADATA.pb | \
    sed -E 's#.*ofl/([a-zA-Z0-9]+)/METADATA.*#\1#g')

# Sitara is broken so get it out
fontFamilies=$(echo $fontFamilies | sed 's/sitara //')

# Show what you got
echo -n "Found" $(echo "$fontFamilies" | wc -w) "latin font families "
echo "out of" $(ls $googleFontDir/ofl/*/METADATA.pb | wc -l)

fonts=""
for i in $fontFamilies
do
    fonts="$fonts $(echo "$googleFontDir"/ofl/$i/*.ttf)"
done
echo "Found" $(echo "$fonts" | wc -w) "fonts"

# Generate 32x32 images of each character a-z for each font
for i in $fonts
do
    fontName=$(basename "$i" .ttf)
    fontDestDir="extra/fonts/$fontName"
    echo "$fontName"
    mkdir -p "$fontDestDir"
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
