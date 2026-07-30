from pathlib import Path
from pydantic import validate_call

@validate_call
def extract_file_extension(filename: str) -> str:
    """
    Dosya adından en sondaki uzantıyı güvenli ve okunur şekilde ayıklar.
    Örn: 'rapor.v1.final.pdf' -> 'pdf'
         'zararli.php.jpg'     -> 'jpg'
    """
    if not filename or "." not in filename:
        return ""
    
    # Path().suffix bize en sondaki uzantıyı noktasıyla verir (Örn: '.pdf')
    extension_with_dot = Path(filename).suffix # sağdan sola nokta bakıyor.
    
    # Noktayı kaldır ve küçük harfe çevir (Örn: 'PDF' -> 'pdf')
    clean_extension = extension_with_dot.lstrip(".").lower()
    
    return clean_extension