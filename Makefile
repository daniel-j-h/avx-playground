include Config.mk

Example: Example.o

watch:
	while ! inotifywait --event modify *.cc; do clear && make; done

clean:
	$(RM) *.o Example

.PHONY: watch clean
