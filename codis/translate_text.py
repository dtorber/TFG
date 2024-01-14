import requests


FROM_LANG = "ca"
TO_LANG = "es"
URL_SERVICE = "http://boso.dsic.upv.es:9009/translate"

lines = [
    "El vicesecretari general del PP assegura que \"en cap moment\" ha observat per part de Rajoy o la resta de la direcci\u00f3 nacional \"res que no sigui suport a la figura de Cifuentes\".",
    "El nou r\u00e0nquing de la plataforma global Lyst, que analitza qu\u00e8 es mou dins del m\u00f3n de la moda, s'ha tornat a fer p\u00fablic despr\u00e9s de difondre's el nou r\u00e0nquing trimestral.",
    "El ministeri p\u00fablic considera Mas autor dels delictes de desobedi\u00e8ncia greu i prevaricaci\u00f3 administrativa, i a Ortega i Rigau les considera cooperadores necess\u00e0ries dels mateixos delictes."
]


elements = [(i, text, FROM_LANG, TO_LANG) for i, text in enumerate(lines)]

translated_elements = requests.post(
    URL_SERVICE, json=elements
).json()

if not isinstance(translated_elements, list):
    raise Exception(
        f"Exception in the translation service: {translated_elements}"
    )

for (i, text, _, to_lang, translation) in translated_elements:
    if translation is not None and len(translation) > 0 and text != translation:
        print(translation)

    else:
        print(f"-- The text {i} was not translated --")
