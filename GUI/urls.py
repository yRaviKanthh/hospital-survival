"""
URL configuration for GUI project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', inclu.0de('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from basics.views import index
from basics.views import training_data

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/',index,name="index"),
    path('training_data/',training_data,name="training_data"),
]
