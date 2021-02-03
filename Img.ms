try (destroydialog ::gvp) catch()

rollout gvp "Print Vertex Point"
(
Global obj, dir_path, outFile, sv
	
button c_path "Path Save"
button c "Stamp Vertex Pos. XY"
	
	
	on c_path pressed do
	(
		dir_path = getSaveFileName caption:"Name Save File" initialDir:(getDir #export) filename:"PositionVertex_XY.csv"
        outFile = createFile dir_path
	)
	
		on c pressed do
	(
		selection.count == 1 and classof selection[1].baseobject == Editable_Poly
			---
		obj = selection[1].baseobject 
		format "Name: %\n\n" (selection[1].name) to:outFile
		format "% % % % %\n\n" ("X") (";") ("Y") (";") ("Z") to:outFile
sv = for i in $.selectedVerts do 
(
	format "% % % % %\n" (i.pos.x) (";") (i.pos.y) (";") (i.pos.z) to:outFile
)
			close outFile
	)
								
)
createdialog gvp style:#(#style_titlebar, #style_sysmenu, #style_toolwindow, #style_sunkenedge)