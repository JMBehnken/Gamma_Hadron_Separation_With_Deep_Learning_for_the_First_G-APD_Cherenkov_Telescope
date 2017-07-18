e.py f.py: d.py
	python d.py
	touch e.py
	touch f.py

a.py:b.py
	python b.py
	touch a.py

b.py:c.py
	python c.py
	touch b.py

d.py: a.py b.py
	python a.py
	python b.py
	touch d.py