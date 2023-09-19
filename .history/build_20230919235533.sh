#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

cd mysite
python manage.py collectstatic --no-input
cd mysite
python manage.py migrate
cd mysite
python manage.py createsuperuser