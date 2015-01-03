name = practical-cnn-2015a
DST = vgg@login.robots.ox.ac.uk:WWW/share
DSTDOC = vgg@login.robots.ox.ac.uk:WWW/practicals/cnn

.PHONY: prepack, pack, pack-data, pack-code, post, clean, distclean

pack-all: pack-data pack-code pack

code=\
decodeCharacters.m \
exercise1.m \
exercise2.m \
exercise3.m \
exercise4.m \
exercise5.m \
extractBlackBlobs.m \
initializeCharacterCNN.m \
setup.m \
tinycnn.m \
README.md \
vlfeat \
matconvnet

doc=\
doc/images \
doc/instructions.html

data=\
data/dots.jpg \
data/sentence-lato.png \
data/charsdb.mat \
data/imagenet-vgg-verydeep-16.mat

tarflags=--exclude='.git*'

code:=$(addprefix $(CURDIR)/,$(code))
data:=$(addprefix $(CURDIR)/,$(data))
deps:=$(shell find $(code) $(doc) $(data) -type f | sed "s/ /\\\\ /g")

pack: data/$(name).tar.gz
pack-data: data/$(name)-data-only.tar.gz
pack-code: data/$(name)-code-only.tar.gz

data/$(name).tar.gz: $(deps)
	rm -rf data/$(name)
	mkdir -p data/$(name)/{doc,data}
	ln -sf $(data) data/$(name)/data/
	ln -sf $(doc) data/$(name)/doc/
	ln -sf $(code) data/$(name)/
	tar -C data -czvhf data/$(name).tar.gz $(tarflags) $(name)/

data/$(name)-data-only.tar.gz: $(deps)
	rm -rf data/$(name)
	mkdir -p data/$(name)/{doc,data}
	ln -sf $(data) data/$(name)/
	tar -C data -czvhf data/$(name)-data-only.tar.gz $(tarflags) $(name)/

data/$(name)-code-only.tar.gz: $(deps)
	rm -rf data/$(name)
	mkdir -p data/$(name)/{doc,data}
	ln -sf $(doc) data/$(name)/doc/
	ln -sf $(code) data/$(name)/
	tar -C data -czvhf data/$(name)-code-only.tar.gz $(tarflags) $(name)/

post-doc:
	rsync -rvt doc/images $(DSTDOC)/
	rsync -vt doc/instructions.html $(DSTDOC)/index.html

post: pack-all
	rsync -vt data/$(name).tar.gz $(DST)/
	rsync -vt data/$(name)-data-only.tar.gz $(DST)/
	rsync -vt data/$(name)-code-only.tar.gz $(DST)/

clean:
	find . -name '*~' -delete

distclean: clean
	rm -f data/$(name)*.tar.gz
