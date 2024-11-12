from .settings import *

# Test-specific settings
DEBUG = True
SECRET_KEY = 'test-key-not-for-production'

# Django Ninja settings
NINJA_PAGINATION_CLASS = 'ninja.pagination.PageNumberPagination'
NINJA_PAGINATION_PER_PAGE = 100
NINJA_PAGINATION_MAX_LIMIT = 1000
NINJA_NUM_PROXIES = None
NINJA_DEFAULT_THROTTLE_RATES = {
    'user': None,
    'anon': None
}

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
    }
}

# Session settings
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
