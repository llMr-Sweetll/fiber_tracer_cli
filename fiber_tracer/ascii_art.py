"""
ASCII Art and Branding Module for Mr. Sweet's Fiber Tracer
Provides animated startup sequences and branded messages.
"""

import sys
import time
import random
from typing import Optional

# Try to import colorama for colored output (optional)
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # Fallback classes if colorama is not available
    class Fore:
        CYAN = YELLOW = GREEN = RED = MAGENTA = BLUE = WHITE = ''
        RESET = ''
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ''


class MrSweetBranding:
    """Handle all branding and ASCII art for Mr. Sweet's Fiber Tracer."""
    
    # ASCII Art Components
    LOGO = """
╔═══════════════════════════════════════╗
║        MR. SWEET'S FIBER TRACER       ║
║      "To infinity and beyond!" 🚀     ║
╚═══════════════════════════════════════╝
    """
    
    LOGO_SIMPLE = """
    ╔════════════════════════╗
    ║      MR. SWEET         ║
    ║    FIBER TRACER        ║
    ╚════════════════════════╝
    """
    
    ROCKET = "🚀"
    SPARKLES = "✨"
    
    # Motivational quotes from Mr. Sweet
    QUOTES = [
        "Every fiber tells a story!",
        "Analyzing composites with style!",
        "Science meets creativity!",
        "Precision in every pixel!",
        "Where fibers meet innovation!",
        "Tracing the future, one fiber at a time!",
        "Making composite analysis sweet!",
    ]
    
    @staticmethod
    def typewriter_effect(text: str, delay: float = 0.05, color: str = Fore.CYAN):
        """Display text with typewriter effect."""
        for char in text:
            sys.stdout.write(color + char)
            sys.stdout.flush()
            time.sleep(delay)
        print(Style.RESET_ALL if COLORS_AVAILABLE else "")
    
    @staticmethod
    def animate_startup(simple: bool = False):
        """Show animated startup sequence."""
        print("\n")
        
        # Step 1: Typewriter reveal of "Mr. Sweet"
        name_parts = ["M", "r", ".", " ", "S", "w", "e", "e", "t"]
        current = ""
        for part in name_parts:
            current += part
            sys.stdout.write(f"\r{Fore.YELLOW if COLORS_AVAILABLE else ''}{current}")
            sys.stdout.flush()
            time.sleep(0.1)
        
        time.sleep(0.5)
        print("\n")
        
        # Step 2: Show the logo with fade-in effect
        logo = MrSweetBranding.LOGO_SIMPLE if simple else MrSweetBranding.LOGO
        
        # Simulate fade-in by printing line by line
        for line in logo.strip().split('\n'):
            print(f"{Fore.CYAN if COLORS_AVAILABLE else ''}{line}")
            time.sleep(0.1)
        
        print(Style.RESET_ALL if COLORS_AVAILABLE else "")
        
        # Step 3: Loading sequence
        print(f"\n{Fore.GREEN if COLORS_AVAILABLE else ''}Initializing Fiber Tracer...")
        time.sleep(0.3)
        
        # Animated loading dots
        for i in range(3):
            for dots in ['   ', '.  ', '.. ', '...']:
                sys.stdout.write(f"\rLoading modules{dots}")
                sys.stdout.flush()
                time.sleep(0.2)
        
        print(f"\rLoading modules... {Fore.GREEN if COLORS_AVAILABLE else ''}✓")
        
        # Step 4: Random quote
        quote = random.choice(MrSweetBranding.QUOTES)
        print(f"\n{Fore.MAGENTA if COLORS_AVAILABLE else ''}💡 {quote}")
        
        print(f"\n{Fore.GREEN if COLORS_AVAILABLE else ''}Ready to trace fibers! {MrSweetBranding.ROCKET}\n")
        print("=" * 45)
    
    @staticmethod
    def show_completion(success: bool = True, fibers_found: Optional[int] = None):
        """Show completion message with branding."""
        print("\n" + "=" * 45)
        
        if success:
            print(f"\n{MrSweetBranding.SPARKLES} {Fore.GREEN if COLORS_AVAILABLE else ''}Analysis Complete! {MrSweetBranding.SPARKLES}")
            
            if fibers_found:
                print(f"{Fore.CYAN if COLORS_AVAILABLE else ''}Found {fibers_found} fibers!")
            
            print(f"\n{Fore.YELLOW if COLORS_AVAILABLE else ''}\"That wasn't flying, that was falling with style!\"")
            print(f"{Fore.MAGENTA if COLORS_AVAILABLE else ''}~ Fibers traced by Mr. Sweet ~\n")
        else:
            print(f"\n{Fore.RED if COLORS_AVAILABLE else ''}Analysis encountered an issue.")
            print(f"{Fore.YELLOW if COLORS_AVAILABLE else ''}\"Even Buzz had to learn to fly!\"")
            print(f"{Fore.CYAN if COLORS_AVAILABLE else ''}~ Check the logs and try again ~\n")
        
        print("=" * 45)
    
    @staticmethod
    def show_progress_message():
        """Show a random progress message."""
        messages = [
            "Analyzing fiber structures...",
            "Detecting fiber orientations...",
            "Calculating tortuosity...",
            "Measuring fiber diameters...",
            "Building 3D volume...",
            "Segmenting fibers with precision...",
            "Extracting fiber properties...",
        ]
        return random.choice(messages)
    
    @staticmethod
    def show_mini_banner():
        """Show a mini banner for quick operations."""
        print(f"\n{Fore.CYAN if COLORS_AVAILABLE else ''}╔═══════════════════════╗")
        print(f"║   MR. SWEET {MrSweetBranding.ROCKET}        ║")
        print(f"╚═══════════════════════╝{Style.RESET_ALL if COLORS_AVAILABLE else ''}\n")
    
    @staticmethod
    def show_error_art():
        """Show error message with style."""
        print(f"""
{Fore.RED if COLORS_AVAILABLE else ''}
    ╔═══════════════════════╗
    ║      OOPS! 😅         ║
    ║   Something went      ║
    ║      sideways!        ║
    ╚═══════════════════════╝
{Style.RESET_ALL if COLORS_AVAILABLE else ''}
        """)
    
    @staticmethod
    def show_test_banner():
        """Special banner for test mode."""
        print(f"""
{Fore.YELLOW if COLORS_AVAILABLE else ''}
    ╔═══════════════════════════════╗
    ║     MR. SWEET'S TEST MODE     ║
    ║   Creating synthetic fibers   ║
    ║         for testing! 🧪        ║
    ╚═══════════════════════════════╝
{Style.RESET_ALL if COLORS_AVAILABLE else ''}
        """)
    
    @staticmethod
    def get_signature():
        """Get Mr. Sweet's signature for files."""
        return """
# ═══════════════════════════════════════
# Created with ❤️ by Mr. Sweet
# "To infinity and beyond!" 🚀
# Contact: hegde.g.chandrashekhar@gmail.com
# ═══════════════════════════════════════
"""


def animate_startup(simple: bool = False, test_mode: bool = False):
    """Convenience function to show startup animation."""
    branding = MrSweetBranding()
    if test_mode:
        branding.show_test_banner()
    else:
        branding.animate_startup(simple=simple)


def show_completion(success: bool = True, fibers_found: Optional[int] = None):
    """Convenience function to show completion message."""
    MrSweetBranding.show_completion(success=success, fibers_found=fibers_found)


def show_mini_banner():
    """Convenience function to show mini banner."""
    MrSweetBranding.show_mini_banner()


# Easter egg function
def buzz_lightyear_mode():
    """Secret Buzz Lightyear mode - activated with --buzz flag."""
    print(f"""
{Fore.MAGENTA if COLORS_AVAILABLE else ''}
    ╔═══════════════════════════════════════╗
    ║         BUZZ LIGHTYEAR MODE!          ║
    ║                                       ║
    ║    "I'm not a flying toy, I'm a      ║
    ║     fiber tracer with style!" 🚀      ║
    ║                                       ║
    ║         SPACE RANGER ALPHA            ║
    ║      FIBER ANALYSIS DIVISION          ║
    ╚═══════════════════════════════════════╝
    
    Star Command says: Let's trace those fibers!
{Style.RESET_ALL if COLORS_AVAILABLE else ''}
    """)


if __name__ == "__main__":
    # Demo the animations
    print("Demo: Startup Animation")
    animate_startup()
    
    print("\nDemo: Completion Message")
    show_completion(success=True, fibers_found=42)
    
    print("\nDemo: Mini Banner")
    show_mini_banner()
    
    print("\nDemo: Test Mode")
    animate_startup(test_mode=True)
