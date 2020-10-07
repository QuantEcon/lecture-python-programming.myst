var menubarDiv = document.createElement("div");

menubarDiv.innerHTML =
'<link rel="stylesheet" href="https://assets.quantecon.org/css/menubar-20200503.css">\n' +
'<div class="qemb"> <!-- QuantEcon menubar -->\n' +
    '<p class="qemb-logo"><a href="https://quantecon.org/" title="quantecon.org"><span class="show-for-sr">QuantEcon</span></a></p>\n' +
    '<div class="qemb-menu">\n' +
        '<ul class="qemb-groups">\n' +
            '<li>\n' +
                '<span>Lectures</span>\n' +
                '<ul>\n' +
                '<li><a href="https://python-programming.quantecon.org/" title="Python Programming for Economics and Finance"><span>Python Programming for Economics and Finance</span></a></li>\n' +
                '<li><a href="https://python.quantecon.org/" title="Quantitative Economics with Python"><span>Quantitative Economics with Python</span></a></li>\n' +
                '<li><a href="https://python-advanced.quantecon.org/" title="Advanced Quantitative Economics with Python"><span>Advanced Quantitative Economics with Python</span></a></li>\n' +
                '<li><a href="https://julia.quantecon.org/" title="Quantitative Economics with Julia"><span>Quantitative Economics with Julia</span></a></li>\n' +
                '<li><a href="https://datascience.quantecon.org/" title="DataScience"><span>QuantEcon DataScience</span></a></li>\n' +
                '<li><a href="http://cheatsheets.quantecon.org/" title="Cheatsheets"><span>Cheatsheets</span></a></li>\n' +
                '</ul>\n' +
            '</li>\n' +
            '<li>\n' +
                '<span>Code</span>\n' +
                '<ul>\n' +
                '<li><a href="https://quantecon.org/quantecon-py" title="QuantEcon.py"><span>QuantEcon.py</span></a></li>\n' +
                '<li><a href="https://quantecon.org/quantecon-jl" title="QuantEcon.jl"><span>QuantEcon.jl</span></a></li>\n' +
                '<li><a href="https://jupinx.quantecon.org/">Jupinx</a></li>\n' +
                '</ul>\n' +
            '</li>\n' +
            '<li>\n' +
                '<span>Notebooks</span>\n' +
                '<ul>\n' +
                '<li><a href="https://quantecon.org/notebooks" title="QuantEcon Notebook Library"><span>NB Library</span></a></li>\n' +
                '<li><a href="http://notes.quantecon.org/" title="QE Notes"><span>QE Notes</span></a></li>\n' +
                '</ul>\n' +
            '</li>\n' +
            '<li>\n' +
                '<span>Community</span>\n' +
                '<ul>\n' +
                '<li><a href="http://blog.quantecon.org/" title="Blog"><span>Blog</span></a></li>\n' +
                '<li><a href="http://discourse.quantecon.org/" title="Forum"><span>Forum</span></a></li>\n' +
                '</ul>\n' +
            '</li>\n' +
        '</ul>\n' +
        '<ul class="qemb-links">\n' +
            '<li><a href="http://store.quantecon.org/" title="Store"><span class="show-for-sr">Store</span></a></li>\n' +
            '<li><a href="https://github.com/QuantEcon/" title="Repository"><span class="show-for-sr">Repository</span></a></li>\n' +
            '<li><a href="https://twitter.com/quantecon" title="Twitter"><span class="show-for-sr">Twitter</span></a></li>\n' +
        '</ul>\n' +
    '</div>\n' +
'</div>';

document.body.prepend(menubarDiv);

var menuLabels = document.querySelectorAll('.qemb-groups>li>span');
var menuLists = document.querySelectorAll('.qemb-groups>li>ul');
function hideMenu(){
    var isClickInside = false;
    for (var i = 0; i < menuLists.length; i++) {
        if (menuLabels[i].contains(event.target)) {
            isClickInside = true;
        }
    }
    if (!isClickInside) {
        for (var i = 0; i < menuLabels.length; i++) {
            menuLabels[i].classList.remove('active');
            menuLists[i].style.display = "";
        }
        document.removeEventListener('click', hideMenu);
    }
}
for (var i = 0; i < menuLabels.length; i++) {
    menuLabels[i].addEventListener('click', function() {
        if ( this.classList.contains('active') ) {
            this.classList.remove('active');
            this.nextElementSibling.style.display = "";
            document.removeEventListener('click', hideMenu);
        } else {
            for (var j = 0; j < menuLabels.length; j++) {
                menuLabels[j].classList.remove('active');
                menuLists[j].style.display = "";
            }
            this.classList.add('active');
            this.nextElementSibling.style.display = "block";
            document.addEventListener('click', hideMenu);
        }
    })
}